import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import os
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (220,220,220), (31,102,156), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
anchor_marker = ['.','.','.','.','.','.','x','x','s','s','s','s','*','*']
uncertainty_threshold = 0.3

class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_GP_classification(X, Y, epoch=100, learning_rate=0.1):
    '''
        Exact GP Regression on Classification Labels
        输入X：样本的特征
        输入Y: 样本的类别标签
        ref: https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/GP_Regression_on_Classification_Labels.html
    '''
    t_X = torch.Tensor(X)
    t_y = torch.Tensor(Y).long().squeeze()
    # initialize likelihood and model
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = DirichletClassificationLikelihood(t_y, learn_additional_noise=True)
    model = DirichletGPModel(t_X, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print("\nStart training Gaussian Process...")
    for i in range(epoch):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(t_X)
        # Calc loss and backprop gradients
        loss = -mll(output, t_y).sum()
        loss.backward()
        if i % 50 == 0:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, epoch, loss.item(),
                model.covar_module.base_kernel.lengthscale.mean().item(),
                model.likelihood.second_noise_covar.noise.mean().item()
            ))
        optimizer.step()
    
    return model, likelihood

def eval_GP_classification(test_x, model, likelihood):
    '''
        根据高斯过程的模型预测test_x的分布，pred_means的维度为[n,...], n表示分类的类别数量    
        # test_x: [n, f_dim]
        # pred_means: [C, n]    C是类别数量
        # pred_var: [C, n]      C是类别数量
    '''
    model.eval()
    likelihood.eval()
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(test_x)
        pred_means = test_dist.loc
        pred_var = test_dist.variance
    
    test_pred_label = pred_means.max(0)[1]
    return test_pred_label, pred_means.numpy(), pred_var.numpy()

def visualize(args, test_x_mat, test_y_mat, pred_means, pred_var, case_study_frame_id, case_study_pos=None, case_study_type=None):
    '''可视化'''
    # 画均值分布
    fig, ax = plt.subplots(1, args.kmeans, figsize = (args.kmeans*6, 5))
    for i in range(args.kmeans):
        im = ax[i].contourf(
            test_x_mat.numpy(), test_y_mat.numpy(), pred_means[i].reshape((test_x_mat.shape[0],test_x_mat.shape[1])), levels=20 #np.linspace(-9, 2, 21)
        )
        fig.colorbar(im, ax=ax[i])
        ax[i].set_title("Class " + str(i), fontsize = 20)
        # 需要画出样本点
        if case_study_pos is not None:  
            for j in range(case_study_pos.shape[0]):
                c = anchor_color[case_study_type[j]]
                hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])
                ax[i].text(case_study_pos[j, 0], case_study_pos[j, 1], 
                        str(case_study_type[j]), 
                        c=hex_c,
                        fontdict={'size': 8},
                        alpha = 0.4)
    # ax[0].legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=6)
    fig.savefig(os.path.join(args.result_path, '../', 'gaussian_process_vis(mean).png'), dpi=600)

    # 画分类面


    # 画方差分布 uncertainty
    case_study_uncertainty = np.zeros(case_study_pos.shape[0])
    fig, ax = plt.subplots(1, args.kmeans, figsize = (args.kmeans*6, 5))
    for i in range(args.kmeans):
        var_mat = pred_var[i].reshape((test_x_mat.shape[0],test_x_mat.shape[1]))
        im = ax[i].contourf(
            test_x_mat.numpy(), test_y_mat.numpy(), var_mat, levels=20 #np.linspace(0, 2, 21)
        )
        fig.colorbar(im, ax=ax[i])
        ax[i].set_title("Class " + str(i), fontsize = 20)
        # 需要画出样本点(注意：每个类别只画出对应类别的样本点，其中uncertainty超过阈值的单独画出来)
        if case_study_pos is not None:  
            for j in range(case_study_pos.shape[0]):
                c = anchor_color[case_study_type[j]]
                if case_study_type[j] != i:
                    continue
                hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])
                # 如果不确定性比较高，把样本颜色变成白色画出来
                _x = min(var_mat.shape[0], max(0, int(case_study_pos[j, 1] * 100)+100))
                _y = min(var_mat.shape[1], max(0, int(case_study_pos[j, 0] * 100)+100))
                if var_mat[_x, _y] > uncertainty_threshold:
                    hex_c = '#%02x%02x%02x' % (255, 255, 255)
                case_study_uncertainty[j] = var_mat[_x, _y]
                ax[i].text(case_study_pos[j, 0], case_study_pos[j, 1], 
                        str(case_study_type[j]), 
                        c=hex_c,
                        fontdict={'size': 8},
                        alpha = 0.4)
    # ax[0].legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=6)
    fig.savefig(os.path.join(args.result_path, '../', 'gaussian_process_vis(var).png'), dpi=600)
    fig.savefig(os.path.join(args.result_path.replace("cluster_results", "case_study"), 'gaussian_process_vis(var)_frame{0}.png'.format(case_study_frame_id+1)), dpi=600)

    return case_study_uncertainty


if __name__ == "__main__":
    print("import my Gaussian Process module.")