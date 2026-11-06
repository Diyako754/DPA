
import torch
import torch.distributed as dist

from gossip.utils import flatten_tensors, unflatten_tensors


class MomentumTracking:
    """
    Momentum Tracking (MT) method:

        m_i^{k+1} = beta m_i^k + grad f_i(x_i^k)
        \tilde{x}_i^k = sum_j w_ij x_j^k
        x_i^{k+1} = \tilde{x}_i^k - lr (m_i^{k+1} - c_i^k)
        c_i^{k+1} = sum_j w_ij (c_j^k - m_j^{k+1}) + m_i^{k+1}

    Paper initialization (optional, via paper_init=True):
        g_i^0 = grad f_i(x_i^0)
        gbar^0 = (1/n) sum_j g_j^0
        m_i^0 = c_i^0 = (g_i^0 - gbar^0) / (1 - beta)
    """

    def __init__(self, model, device, beta=0.9, weight_decay=0.0, paper_init=False):
        """
        Args:
            model: GossipDataParallel wrapper
            device: torch.device
            beta: momentum coefficient
            weight_decay: L2 weight decay applied as grad += wd * x
            paper_init: whether to use the paper's initialization at the first step
        """
        self.model = model
        self.device = device
        self.beta = float(beta)
        self.weight_decay = float(weight_decay)
        self.paper_init = bool(paper_init)

        if self.paper_init and not (0.0 <= self.beta < 1.0):
            raise ValueError("paper_init requires 0 <= beta < 1 for (1-beta) denominator.")

        # Internal state (lists of tensors matching parameter shapes)
        self.m = None
        self.c = None

        # Lazy init flags
        self._buffers_initialized = False
        self._paper_initialized = False

    def _init_buffers_if_needed(self):
        if self._buffers_initialized:
            return

        with torch.no_grad():
            params = list(self.model.module.parameters())
            self.m = [torch.zeros_like(p.data, device=p.data.device) for p in params]
            self.c = [torch.zeros_like(p.data, device=p.data.device) for p in params]

        self._buffers_initialized = True

    def _get_grads_list(self):
        """
        Read local gradients g_i^k (after loss.backward()).
        Applies weight decay as g += wd * x if weight_decay > 0.
        """
        grads = []
        for p in self.model.module.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p.data))
            else:
                g = p.grad.data
                if self.weight_decay != 0.0:
                    g = g.add(p.data, alpha=self.weight_decay)
                grads.append(g)
        return grads

    def _paper_init_from_current_grads(self, grads):
        """
        Implements paper initialization using current grads as g_i^0.
        Requires dist initialized (world_size = n).
        """
        if not dist.is_initialized():
            raise RuntimeError("paper_init=True requires torch.distributed to be initialized.")

        world_size = dist.get_world_size()

        # Flatten local g_i^0
        g_flat = flatten_tensors(grads).detach()

        # All-reduce to get sum of grads across nodes
        g_sum = g_flat.clone()
        dist.all_reduce(g_sum, op=dist.ReduceOp.SUM)
        g_bar = g_sum.div_(float(world_size))

        # Unflatten to list
        g_bar_list = list(unflatten_tensors(g_bar, grads))

        # m_i^0 = c_i^0 = (g_i^0 - gbar^0)/(1-beta)
        denom = (1.0 - self.beta)
        with torch.no_grad():
            for idx in range(len(grads)):
                init_val = grads[idx].sub(g_bar_list[idx]).div(denom)
                self.m[idx].copy_(init_val)
                self.c[idx].copy_(init_val)

        self._paper_initialized = True

    @torch.no_grad()
    def step_from_grads(self, lr: float):
        """
        One MT iteration using the gradients currently stored in p.grad.

        Returns:
            data_transferred_bytes: total bytes transferred by the TWO gossip mixes:
                - gossip on x
                - gossip on (c - m)
        """
        self._init_buffers_if_needed()

        # g_i^k from current backward
        grads = self._get_grads_list()

        # Paper initialization at the FIRST call (k = 0), using current grads as g^0
        if self.paper_init and not self._paper_initialized:
            self._paper_init_from_current_grads(grads)

        # Save c_i^k for the primal update and for r = (c^k - m^{k+1})
        c_old = [t.clone() for t in self.c]

        # Momentum update: m^{k+1} = beta m^k + g^k
        for i in range(len(self.m)):
            self.m[i].mul_(self.beta).add_(grads[i])

        # 1) Gossip on x: \tilde{x} = W x
        dt_x = self.model.gossip_average_parameters_inplace()

        # Primal update: x^{k+1} = \tilde{x} - lr (m^{k+1} - c^k)
        for p, mi, ci_old in zip(self.model.module.parameters(), self.m, c_old):
            direction = mi.sub(ci_old)  # (m^{k+1} - c^k)
            p.data.add_(direction, alpha=-float(lr))

        # 2) Gossip on r = (c^k - m^{k+1}):
        r_list = [ci_old.sub(mi) for ci_old, mi in zip(c_old, self.m)]
        dt_r = self.model.gossip_average_tensor_list_inplace(r_list)

        # Tracking update: c^{k+1} = W(c^k - m^{k+1}) + m^{k+1} = W r + m^{k+1}
        for i in range(len(self.c)):
            self.c[i].copy_(r_list[i]).add_(self.m[i])

        # Clear grads (trainer also calls optimizer.zero_grad(), but safe here too)
        for p in self.model.module.parameters():
            p.grad = None

        return float(dt_x + dt_r)
