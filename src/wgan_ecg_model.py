import torch

def gradient_penalty(critic, real, fake):
    batch_size = real.size(0)

    epsilon = torch.rand(batch_size, 1)
    epsilon = epsilon.expand_as(real)

    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(batch_size, -1)
    gradient_norm = gradient.norm(2, dim=1)

    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp