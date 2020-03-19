from time import time

import torch
from FrEIA.framework import *
from FrEIA.modules import *

import monitoring
loss_labels = ['-log p(z)', '-log |det(J)|']


# Load training configuration, model and data set

# from configs.fourier_curve.conditional_cinn_1 import c, model, model_inverse
# from configs.fourier_curve.conditional_cinn_2 import c, model, model_inverse
# from configs.fourier_curve.conditional_cinn_4 import c, model, model_inverse
# from configs.fourier_curve.conditional_cinn_8 import c, model, model_inverse
# from configs.fourier_curve.conditional_hint_1_full import c, model, model_inverse
# from configs.fourier_curve.conditional_hint_2_full import c, model, model_inverse
# from configs.fourier_curve.conditional_hint_4_full import c, model, model_inverse
from configs.fourier_curve.conditional_hint_8_full import c, model, model_inverse

# from configs.plus_shape.conditional_cinn_4 import c, model, model_inverse
# from configs.plus_shape.conditional_cinn_8 import c, model, model_inverse
# from configs.plus_shape.conditional_hint_4_full import c, model, model_inverse
# from configs.plus_shape.conditional_hint_8_full import c, model, model_inverse


# Init trainable model parameters
params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
if c.init_scale > 0:
    for p in params_trainable:
        p.data = c.init_scale * torch.randn_like(p.data)
# Count total number of trainable parameters
n_model_params = sum([p.numel() for p in params_trainable])
print(f'\nModel {c.suffix} has {n_model_params:,} trainable parameters.\n')
# assert False


# Prepare optimizer and learning rate schedule
optim = torch.optim.Adam(params_trainable, lr=c.lr_init,
                         betas=c.adam_betas, eps=1e-4,
                         weight_decay=c.l2_weight_reg)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1,
                                               gamma=(c.final_decay)**(1./c.n_epochs))

# For visualization
vis_batchsize = 300
vis_y = torch.Tensor([c.vis_y_target]*vis_batchsize).to(c.device)
vis_latent = torch.randn(vis_batchsize, c.ndim_x).to(c.device)


def save(name):
    torch.save({'opt':optim.state_dict(),
                'net':model.state_dict()}, name)

def load(name):
    state_dicts = torch.load(name)
    model.load_state_dict(state_dicts['net'])
    try:
        optim.load_state_dict(state_dicts['opt'])
    except ValueError:
        print('Cannot load optimizer for some reason or other')

def save_sample(y=c.vis_y_target, N=500, modelpath=f'output/{c.suffix}.pt'):
    # create sample from saved model
    if modelpath:
        load(modelpath)
    model.eval()
    with torch.no_grad():
        z_sample = torch.cuda.FloatTensor(N, c.ndim_z).normal_()
        y_sample = torch.tensor(y).expand(N, c.ndim_y).cuda()
        x_sample = model_inverse(y_sample, z_sample)
        np.save(f'output/samples/{c.suffix}_sample-{N}', x_sample.cpu())


def x_jac():
    jacobian = 0
    for node in model.node_list:
        if 'hac_x' in node.name or 'ac_y_to_x' in node.name:
            jacobian += node.module.jacobian(None)
    return jacobian


def train_epoch(i_epoch, test=False):

    if not test:
        model.train()
        loader = c.train_loader

    if test:
        model.eval()
        loader = c.test_loader
        nograd = torch.no_grad()
        nograd.__enter__()
        nll_x = []


    batch_idx = 0
    loss_history = []

    for x, y in loader:
        optim.zero_grad()
        batch_losses = []
        batch_idx += 1
        if batch_idx > c.max_batches_per_epoch > 0: break

        x, y = x.to(c.device), y.to(c.device)
        x += 0.01 * torch.randn_like(x)

        # Forward pass
        if 'hint' in c.suffix:
            z_y, z_x = model([y, x])
            z = torch.cat([z_x, z_y], dim=-1)
            log_jacobian = model.log_jacobian([y, x], run_forward=False)
            if test:
                nll_x.append((0.5 * torch.sum(z_x**2, dim=1).mean() - x_jac().mean()).item())
        elif 'cinn' in c.suffix:
            z = model([x], c=[y])
            log_jacobian = model.log_jacobian([x], c=[y], run_forward=False)

        # Maximum likelihood loss terms
        batch_losses.append(0.5 * torch.sum(z**2, dim=1).mean())
        batch_losses.append(-log_jacobian.mean())

        # Add up all losses
        loss_total = sum(batch_losses)
        loss_history.append([l.item() for l in batch_losses])

        # Compute gradients
        if not test:
            loss_total.backward()

            # Clamp gradients
            for p in params_trainable:
                p.grad.data.clamp_(-5.00, 5.00)

            # Parameter update
            optim.step()

            # Update progress bar
            monitoring.visualizer.update_progress(batch_idx, i_epoch+1)

    if test:
        # Update plots
        latent_sample = z[:500,:].data.cpu().numpy()
        with torch.no_grad():
            vis_x = model_inverse(vis_y, vis_latent).data.cpu().numpy()
        monitoring.visualizer.update_plots(latent_sample, vis_x)
        # Output negative log-likelihood and bits/dim of x lane (only for HINT)
        if len(nll_x) > 0:
            print('      NLL_x:', np.mean(nll_x), ' |   bits/dim:', -np.mean(nll_x) / (c.ndim_x * np.log(2.)))

        nograd.__exit__(None, None, None)

    return np.mean(loss_history, axis=0)


def main():
    monitoring.restart(c, loss_labels)
    monitoring.visualizer.print_config()

    t_start = time()
    try:
        for i_epoch in range(c.n_epochs):
            if i_epoch < c.pre_low_lr:
                for param_group in optim.param_groups:
                    param_group['lr'] = c.lr_init * 3e-2

            train_losses = train_epoch(i_epoch)
            test_losses  = train_epoch(i_epoch, test=True)

            monitoring.visualizer.update_losses(np.concatenate([train_losses, test_losses]),
                                                lr_scheduler.get_lr(), logscale=False)

            lr_scheduler.step()
        save(f'output/{c.suffix}.pt')
    except:
        save(f'output/{c.suffix}.pt' + '_ABORT')
        raise

    finally:
        print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))


if __name__ == "__main__":
    main()
    # save_sample()
