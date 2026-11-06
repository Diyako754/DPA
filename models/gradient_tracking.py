
import torch


class GradientTracking:
    """
    Plain Gradient Tracking (GT) adapted to the current package setup.

    Standard GT form:
        x_i^{k+1} = sum_j w_ij x_j^k - lr * y_i^k
        y_i^k     = sum_j w_ij y_j^{k-1} + g_i^k - g_i^{k-1}

    where
        g_i^k = grad f_i(x_i^k) [+ weight decay term if enabled]

    Initialization:
        y_i^0 = g_i^0

    Notes:
      - This implementation uses the gradient currently stored in p.grad
        as g_i^k.
      - On the first call, it initializes y = g^0 and performs:
            x^1 = W x^0 - lr * y^0
      - On later calls, with current grad g^k available, it updates:
            y^k = W y^{k-1} + g^k - g^{k-1}
        and then:
            x^{k+1} = W x^k - lr * y^k

    This class assumes the wrapped GossipDataParallel model provides:
      - gossip_average_parameters_inplace()
      - gossip_average_tensor_list_inplace(tensor_list)
    """

    def __init__(self, model, device, beta=0.0, weight_decay=0.001):
        """
        Args:
            model: GossipDataParallel wrapper
            device: torch.device
            beta: accepted only for compatibility with trainer.py; unused in plain GT
            weight_decay: L2 penalty added as grad += wd * x
        """
        self.model = model
        self.device = device
        self.weight_decay = float(weight_decay)

        # GT state
        self.y = None
        self.prev_grad = None

        self._buffers_initialized = False
        self._gt_initialized = False

    def _params(self):
        return [p for p in self.model.module.parameters() if p.requires_grad]

    def _init_buffers_if_needed(self):
        if self._buffers_initialized:
            return

        params = self._params()
        self.y = [torch.zeros_like(p.data, device=p.data.device) for p in params]
        self.prev_grad = [torch.zeros_like(p.data, device=p.data.device) for p in params]
        self._buffers_initialized = True

    def _get_grads_list(self):
        grads = []
        for p in self._params():
            if p.grad is None:
                g = torch.zeros_like(p.data)
            else:
                g = p.grad.data.clone()
                if self.weight_decay != 0.0:
                    g.add_(p.data, alpha=self.weight_decay)
            grads.append(g)
        return grads

    @torch.no_grad()
    def step_from_grads(self, lr: float):
        """
        One plain-GT iteration using gradients currently stored in p.grad.

        Returns:
            data_transferred_bytes: total bytes transferred by:
                - gossip on y
                - gossip on x
        """
        self._init_buffers_if_needed()

        params = self._params()
        grads = self._get_grads_list()

        data_transferred = 0.0

        if not self._gt_initialized:
            # Initialization: y^0 = g^0
            for y_i, g_i, pg_i in zip(self.y, grads, self.prev_grad):
                y_i.copy_(g_i)
                pg_i.copy_(g_i)
            self._gt_initialized = True
        else:
            # y^k = W y^{k-1} + g^k - g^{k-1}
            dt_y = self.model.gossip_average_tensor_list_inplace(self.y)
            data_transferred += float(dt_y)

            for y_i, g_i, pg_i in zip(self.y, grads, self.prev_grad):
                y_i.add_(g_i).sub_(pg_i)
                pg_i.copy_(g_i)

        # x^{k+1} = W x^k - lr * y^k
        dt_x = self.model.gossip_average_parameters_inplace()
        data_transferred += float(dt_x)

        for p, y_i in zip(params, self.y):
            p.data.add_(y_i, alpha=-float(lr))

        for p in params:
            p.grad = None

        return float(data_transferred)