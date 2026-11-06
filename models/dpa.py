import torch
from .utils import flatten_tensors, unflatten_tensors


class DPA:
    """
    Decentralized Primal Averaging.

    Modes:
      T_gossip_x == 0:
        DPA
        - Every iteration:
            gossip y
            gossip x

      T_gossip_x > 0:
        DPA-1G
        - Normal iterations:
            gossip y
            update x locally
        - Every T_gossip_x iterations:
            skip gradient
            freeze y and u
            gossip x

    Variables:
      x: live model parameters
      y: optimization sequence
      u: local quasi-global momentum buffer

    Updates:
      z_k = b_{k+1} y_k + (1 - b_{k+1}) x_k
      g_k = grad(z_k)
      u_hat_k = beta_1 * u_k + g_k
      y_{k+1} = W(y_k - lr * u_hat_k)
      d_{k+1} = (y_k - y_{k+1}) / lr
      u_{k+1} = (1 - beta_2) * u_k + beta_2 * d_{k+1}
    """

    def __init__(
        self,
        model,
        device,
        beta_1=0.9,
        beta_2=1.0,
        warmup_steps: int = 0,
        a_start: float = 0.3,
        a_end: float = 0.1,
        a_steps: int = 2000,
        a_schedule: str = "linear",
        qa: float = 1.0,
        b_start: float = 0.9,
        b_end: float = 0.05,
        b_steps: int = 2000,
        b_schedule: str = "linear",
        qb: float = 1.0,
        T_gossip_x: int = 0,
    ):
        self.model = model
        self.device = device

        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)

        if not (0.0 <= self.beta_1 < 1.0):
            raise ValueError(f"beta_1 must be in [0, 1), got {self.beta_1}")

        if not (0.0 <= self.beta_2 <= 1.0):
            raise ValueError(f"beta_2 must be in [0, 1], got {self.beta_2}")

        self.warmup_steps = int(warmup_steps)

        self.a_start = float(a_start)
        self.a_end = float(a_end)
        self.a_steps = int(a_steps)
        self.a_schedule = str(a_schedule).lower()
        self.qa = float(qa)

        if self.a_schedule not in ("linear", "exp"):
            raise ValueError(f"a_schedule must be 'linear' or 'exp', got {self.a_schedule}")
        if not (0.0 <= self.a_start <= 1.0):
            raise ValueError(f"a_start must be in [0, 1], got {self.a_start}")
        if not (0.0 <= self.a_end <= 1.0):
            raise ValueError(f"a_end must be in [0, 1], got {self.a_end}")
        if self.qa <= 0.0:
            raise ValueError(f"qa must be > 0, got {self.qa}")
        if self.a_schedule == "exp" and (self.a_start <= 0.0 or self.a_end <= 0.0):
            raise ValueError("Exponential a schedule requires a_start > 0 and a_end > 0.")

        self.b_start = float(b_start)
        self.b_end = float(b_end)
        self.b_steps = int(b_steps)
        self.b_schedule = str(b_schedule).lower()
        self.qb = float(qb)

        if self.b_schedule not in ("linear", "exp"):
            raise ValueError(f"b_schedule must be 'linear' or 'exp', got {self.b_schedule}")
        if not (0.0 <= self.b_start <= 1.0):
            raise ValueError(f"b_start must be in [0, 1], got {self.b_start}")
        if not (0.0 <= self.b_end <= 1.0):
            raise ValueError(f"b_end must be in [0, 1], got {self.b_end}")
        if self.qb <= 0.0:
            raise ValueError(f"qb must be > 0, got {self.qb}")
        if self.b_schedule == "exp" and (self.b_start <= 0.0 or self.b_end <= 0.0):
            raise ValueError("Exponential b schedule requires b_start > 0 and b_end > 0.")

        self.T_gossip_x = int(T_gossip_x)
        if self.T_gossip_x < 0:
            raise ValueError("T_gossip_x must be 0 for DPA or > 0 for DPA-1G.")

        self.req_params = [p for p in self.model.module.parameters() if p.requires_grad]

        self.y = [p.data.detach().clone().to(self.device) for p in self.req_params]
        self.u = [torch.zeros_like(p.data, device=self.device) for p in self.req_params]

        self.t = 0
        self._a_cached = None

        mw = self.model.mixing_weights["self"]
        self.self_w = float(mw.item() if hasattr(mw, "item") else mw)

    def _progress(self, steps: int) -> float:
        if int(steps) <= 0:
            return 1.0
        return float(min(1.0, float(self.t + 1) / float(steps)))

    def _compute_a(self) -> float:
        s = self._progress(self.a_steps)

        if self.a_schedule == "linear":
            a = self.a_start + (self.a_end - self.a_start) * s
        elif self.a_schedule == "exp":
            a = self.a_start * ((self.a_end / self.a_start) ** (s ** self.qa))
        else:
            raise RuntimeError(f"Unknown a_schedule: {self.a_schedule}")

        return float(a)

    def _compute_b(self) -> float:
        s = self._progress(self.b_steps)

        if self.b_schedule == "linear":
            b = self.b_start + (self.b_end - self.b_start) * s
        elif self.b_schedule == "exp":
            b = self.b_start * ((self.b_end / self.b_start) ** (s ** self.qb))
        else:
            raise RuntimeError(f"Unknown b_schedule: {self.b_schedule}")

        return float(b)

    def begin_step(self, eta_k: float):
        _ = eta_k
        a_k1 = self._compute_a()
        b_k1 = self._compute_b()
        self._a_cached = float(a_k1)
        return float(a_k1), float(b_k1)

    def is_x_gossip_step(self) -> bool:
        if self.T_gossip_x <= 0:
            return False
        return ((self.t + 1) % self.T_gossip_x) == 0

    @torch.no_grad()
    def _mix_flat_vector(self, flat_vec):
        flat_vec = flat_vec.to(self.device)
        flat_orig = flat_vec.clone()
        in_msg, data_amt = self.model.gossiper.mix(flat_vec)
        mixed = in_msg + (self.self_w * flat_orig)
        return mixed, data_amt

    @torch.no_grad()
    def make_z_and_load(self, b_k1: float):
        b_k1 = float(b_k1)

        if not (0.0 <= b_k1 <= 1.0):
            raise ValueError(f"b_k1 must be in [0, 1], got {b_k1}")

        saved_x = []

        for idx, p in enumerate(self.req_params):
            x = p.data
            saved_x.append(x.detach().clone())
            z = b_k1 * self.y[idx] + (1.0 - b_k1) * x
            p.data.copy_(z)

        return saved_x

    @torch.no_grad()
    def restore_x(self, saved_x):
        for p, x_saved in zip(self.req_params, saved_x):
            p.data.copy_(x_saved)

    @torch.no_grad()
    def update_y_from_grads(self, lr: float, do_mix: bool = True):
        if not do_mix:
            raise ValueError("DPA requires do_mix=True for the y update.")

        lr = float(lr)
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")

        y_old = [yy.detach().clone() for yy in self.y]
        u_old = [uu.detach().clone() for uu in self.u]

        u_hat_list = []

        for idx, p in enumerate(self.req_params):
            if p.grad is None:
                raise RuntimeError("DPA: p.grad is None. Did you run backward at z_k?")
            u_hat = self.beta_1 * self.u[idx] + p.grad.data
            u_hat_list.append(u_hat)

        tilde_y_list = []

        for idx in range(len(self.y)):
            tilde_y_list.append(self.y[idx] - lr * u_hat_list[idx])

        tilde_y_flat = flatten_tensors(tilde_y_list).to(self.device)
        y_next_flat, data_amt = self._mix_flat_vector(tilde_y_flat)
        y_next_list = unflatten_tensors(y_next_flat, tilde_y_list)

        for idx in range(len(self.y)):
            self.y[idx].copy_(y_next_list[idx])

        d_next = []

        for idx in range(len(self.u)):
            d_k1 = (y_old[idx] - self.y[idx]) / lr
            d_next.append(d_k1)

        for idx in range(len(self.u)):
            self.u[idx].copy_((1.0 - self.beta_2) * u_old[idx] + self.beta_2 * d_next[idx])

        return float(data_amt)

    @torch.no_grad()
    def update_x_from_y(self, a_k1: float = None, do_mix: bool = True):
        if a_k1 is None:
            if self._a_cached is None:
                raise RuntimeError("DPA: a_k1 is missing. Call begin_step first.")
            a = float(self._a_cached)
        else:
            a = float(a_k1)

        if not (0.0 <= a <= 1.0):
            raise ValueError(f"a_k1 must be in [0, 1], got {a}")

        tilde_x_list = []

        for idx, p in enumerate(self.req_params):
            x = p.data
            tx = (1.0 - a) * x + a * self.y[idx]
            tilde_x_list.append(tx)

        if do_mix:
            tilde_x_flat = flatten_tensors(tilde_x_list).to(self.device)
            x_next_flat, data_amt = self._mix_flat_vector(tilde_x_flat)
            x_next_list = unflatten_tensors(x_next_flat, tilde_x_list)

            for p, x_next in zip(self.req_params, x_next_list):
                p.data.copy_(x_next)
        else:
            for p, tx in zip(self.req_params, tilde_x_list):
                p.data.copy_(tx)
            data_amt = 0.0

        self.t += 1
        self._a_cached = None

        return float(a), float(data_amt)

    @torch.no_grad()
    def special_x_gossip_step(self, a_k1: float = None):
        if self.T_gossip_x <= 0:
            raise RuntimeError("special_x_gossip_step is only valid for DPA-1G.")

        if a_k1 is None:
            if self._a_cached is None:
                raise RuntimeError("DPA: a_k1 is missing. Call begin_step first.")
            a = float(self._a_cached)
        else:
            a = float(a_k1)

        if not (0.0 <= a <= 1.0):
            raise ValueError(f"a_k1 must be in [0, 1], got {a}")

        tilde_x_list = []

        for idx, p in enumerate(self.req_params):
            x = p.data
            tx = (1.0 - a) * x + a * self.y[idx]
            tilde_x_list.append(tx)

        tilde_x_flat = flatten_tensors(tilde_x_list).to(self.device)
        x_next_flat, data_amt = self._mix_flat_vector(tilde_x_flat)
        x_next_list = unflatten_tensors(x_next_flat, tilde_x_list)

        for p, x_next in zip(self.req_params, x_next_list):
            p.data.copy_(x_next)

        self.t += 1
        self._a_cached = None

        return float(a), float(data_amt)

    @torch.no_grad()
    def zero_grads(self):
        for p in self.req_params:
            p.grad = None

    @torch.no_grad()
    def load_x_into_model(self):
        return None

    @torch.no_grad()
    def restore_from_saved(self, saved_params):
        return