from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

import numpy as np
import torch
import argparse
import os
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import logsigmoid
from torch.autograd import grad

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import adjusted_rand_score

from classify import Classifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx


tl = np.loadtxt('email-Eu-core-department-labels.txt')
data = np.loadtxt('email-Eu-core.txt', dtype=int)
num_vertices = len(np.unique(data))
adjacency_matrix = torch.zeros(num_vertices, num_vertices)
for r in range(len(data)):
    i = data[r][0]
    j = data[r][1]
    adjacency_matrix[i][j] = 1

num_observed = num_vertices ** 2
observed_edge = torch.zeros(num_observed, 2)
count = 0
for i in range(num_vertices):
    for j in range(num_vertices):
        observed_edge[count, 0] = i
        observed_edge[count, 1] = j
        count = count + 1

edges = adjacency_matrix[(observed_edge[:, 0].long(), observed_edge[:, 1].long())]
vertices = observed_edge[:, 0]
targets = observed_edge[:, 1]


class _RequiredParameter(object):
    def __repr__(self):
        return "<required parameter>"
required = _RequiredParameter()


class SGHMC(torch.optim.Optimizer): 
    def __init__(self, params, lr=required, fraction=None, temperature=1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if fraction is not None:
            if fraction < 0.0:
                raise ValueError("Invalid fraction value: {}".format(fraction))
        else:
            fraction = 1 / lr
        if temperature < 0.0:
            raise ValueError("Invalid temperature value: {}".format(temperature))
        defaults = dict(lr=lr, fraction=fraction, temperature=temperature)
        super(SGHMC, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            fraction = group['fraction']
            temperature = group['temperature']
            lr = group['lr']
            scale = np.sqrt(2.0 * lr * fraction * temperature)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state: # v_{k,1}
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    buf.mul_(lr)
                else: # v_{k,l} l=2,...
                    buf = param_state['momentum_buffer']
                    buf.mul_(1 - lr * fraction).add_(d_p.mul(lr))
                buf.add_(torch.ones_like(buf).normal_().mul(scale))
                p.data.add_(buf.mul(lr))
        return loss


class AdamSGDWeighted(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(AdamSGDWeighted, self).__init__(params, defaults)
   
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.clone().detach()
                if grad.is_sparse:
                    raise RuntimeError('AdamSGDWeighted does not support sparse gradients')
                d_p_adam = self.adam_step(grad, group, p)
                megred_d_p = group['lr'] * grad + group['lr'] * d_p_adam
                p.data.add_(megred_d_p)
                del grad, d_p_adam, megred_d_p
        return loss

    def adam_step(self, grad, group, p):
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        d_p = exp_avg / exp_avg_sq.add_(group['eps']).sqrt()
        return d_p


class Net(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim):
        super(Net, self).__init__()
        self.num_hidden = num_hidden
        self.fc = nn.Linear(input_dim, hidden_dim[0])
        self.fc_list = []
        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.add_module('fc' + str(i + 2), self.fc_list[-1])
        self.logistic = nn.Linear(hidden_dim[-1], output_dim)
        self.add_module('logistic', self.logistic)

    def forward(self, embeddings, targets=None):
        x = torch.sigmoid(self.fc(embeddings))
        for i in range(self.num_hidden - 1):
            x = torch.sigmoid(self.fc_list[i](x))
        if targets is not None:
            weight = self.logistic.weight[targets,:]
            bias = self.logistic.bias[targets]
            x = (x * weight).sum(1) + bias
        else:
            weight = self.logistic.weight
            bias = self.logistic.bias
            x = x.matmul(weight.t())+bias
        return x


def evaluate_embedding_sup(embeddings):
    X = [int(i) for i in tl[:,0]]
    Y = [str(i) for i in tl[:,1]]
    for i in range(1,10):
        tr_frac = i/10.0
        print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(solver='liblinear'))
        clf.split_train_evaluate(X, Y, tr_frac)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

z_dim = 100
net_embedding = nn.Embedding(num_vertices, z_dim).to(device)
net_embedding.weight.data = net_embedding.weight.data.uniform_(-.5, .5) / z_dim 
print(torch.cuda.memory_allocated())

input_dim = z_dim
output_dim = num_vertices
hidden_dim = [200]
num_hidden = len(hidden_dim)
net = Net(num_hidden, hidden_dim, input_dim, output_dim).to(device)
print(torch.cuda.memory_allocated())


loss_func = nn.BCELoss()
def line_loss(e_i, e_j, sign):
    return torch.sum(logsigmoid((e_i*e_j).sum(axis=1) * sign))



############
# pretrain #
############
subn = 500
num_HMC_step = 15
optim_embedding = SGHMC(net_embedding.parameters(), lr=0.03, fraction=10.0, temperature=1e-6) 
optim_net = AdamSGDWeighted(net.parameters(), lr=8e-4)

auc_path = []
recall_path = []
precision_path = []

for epoch in range(301):
    subsample = np.arange(num_observed)
    subsample = []
    for i in range(num_vertices):
        temp = list(range(i*num_vertices, (i+1)*num_vertices))
        np.random.shuffle(temp)
        subsample += temp[:subn]
    subsample = torch.LongTensor(subsample)
    vertices_sub = vertices[subsample.long()]
    targets_sub = targets[subsample.long()]
    edges_sub = adjacency_matrix[vertices[subsample.long()].long(), targets[subsample.long()].long()]
    
    v_i = vertices_sub.long()
    v_j = targets_sub.long()
    sign = (edges_sub-0.5)*2
     
    for repeat in range(num_HMC_step):
        e_i = net_embedding(v_i.to(device))
        e_j = net_embedding(v_j.to(device))
        log_likelihood = line_loss(e_i, e_j, sign.to(device))
        tmp = - loss_func(torch.sigmoid(net(net_embedding(vertices_sub.to(device).long()), targets_sub.to(device).long())), edges_sub.to(device)) * num_observed
        
        log_likelihood = 0.01*log_likelihood + tmp 
        optim_embedding.zero_grad()
        log_likelihood.backward()
        optim_embedding.step()
        
        del e_i,e_j
        del log_likelihood,tmp
        
    optim_net.zero_grad()
    loss = - loss_func(torch.sigmoid(net(net_embedding(vertices_sub.to(device).long()), targets_sub.to(device).long())), edges_sub.to(device)) * num_observed
    loss.backward()
    optim_net.step()
    del loss
    
    if epoch%10==0:
        with torch.no_grad():
            output = torch.zeros(num_vertices, num_vertices)
            for i in range(num_vertices):
                embed_tmp = net_embedding(torch.Tensor([i]).repeat(num_vertices).to(device).long())
                tmp_output = net(embed_tmp, torch.Tensor(np.arange(num_vertices)).to(device).long())
                output[i,:] = tmp_output.cpu().detach()
                del tmp_output
                del embed_tmp
            output = output.reshape((-1,))
            prediction = output > 0
            accuracy = prediction.eq(edges.data.byte()).sum().item() / num_observed
            target_true = torch.sum(edges.data == 1).float()
            predicted_true = torch.sum(prediction == 1).float()
            correct_true = torch.sum((prediction == 1) * (edges.data == 1)).float()
            recall = correct_true / target_true
            precision = correct_true / predicted_true
            recall_path.append(recall)
            precision_path.append(precision)
            prob = torch.sigmoid(output)
            auc = roc_auc_score(edges, prob) 
            loss = loss_func(prob, edges)
            auc_path.append(auc)
            print('epoch: ', epoch, 'loss: ', loss, 'auc: ', auc, 'precision: ', precision, 'recall: ', recall)
            print(evaluate_embedding_sup(net_embedding(vertices.unique().to(device).long()).cpu().detach().numpy()))

torch.save(net.state_dict(), 'email_net.pth')
torch.save(net_embedding.state_dict(), 'email_embed.pth')



#######################
# dimension selection #
#######################
def cal_dev():
    w1 = net.fc.weight.data.clone() 
    w2 = net.logistic.weight.data.clone()
    g_i_new_new = torch.zeros((num_vertices, z_dim))
    for i in range(num_vertices):
        embed_tmp = net_embedding(torch.Tensor([i]).repeat(num_vertices).to(device).long())#.detach()
        tmp_output = torch.sigmoid(net(embed_tmp, torch.Tensor(np.arange(num_vertices)).to(device).long()))#.detach()
        y = torch.sigmoid(net.fc(embed_tmp[0]))
        d1 = torch.diag(y*(1-y)).detach()
        d2 = torch.diag(tmp_output*(1-tmp_output)).detach()
        g_i_new_new += torch.matmul(torch.matmul(torch.matmul(d2, w2), d1), w1).detach().cpu()
    del w1,w2,embed_tmp,tmp_output,y,d1,d2
    return ((g_i_new_new)**2).sum(axis=0)/(num_vertices**2)


class newAdamSGDWeighted(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, lr_sparse=1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, lr_sparse=lr_sparse)
        super(newAdamSGDWeighted, self).__init__(params, defaults)
   
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        cnt = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if cnt==0:
                    temp = - 0.001*torch.sign(p) - np.sqrt(p.shape[0]) * torch.stack([x/torch.sqrt(torch.sum(x**2)) for x in torch.unbind(p, dim=1)], dim=1)
                elif cnt==1:
                    temp = - 0.001*torch.sign(p)
                else: 
                    temp = - torch.sign(p)
                grad = (p.grad + group['lr_sparse'] * temp).clone().detach()
                if grad.is_sparse:
                    raise RuntimeError('AdamSGDWeighted does not support sparse gradients')
                d_p_adam = self.adam_step(grad, group, p)
                megred_d_p = group['lr'] * grad + group['lr'] * d_p_adam
                p.data.add_(megred_d_p)
                del grad, temp, d_p_adam, megred_d_p
                cnt += 1
        return loss

    def adam_step(self, grad, group, p):
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        d_p = exp_avg / exp_avg_sq.add_(group['eps']).sqrt()
        return d_p
    
auc_path = []
recall_path = []
precision_path = []
sparse_path_new = []
ans = cal_dev()
sparse_path_new.append(np.array(ans))    

optim_embedding = SGHMC(net_embedding.parameters(), lr=3e-02, fraction=10.0, temperature=1e-6) 
optim_net_new = newAdamSGDWeighted(net.parameters(), lr=8e-4, lr_sparse=1.0) 

for epoch in range(1,1001):    
    subsample = np.arange(num_observed)
    subsample = []
    for i in range(num_vertices):
        temp = list(range(i*num_vertices, (i+1)*num_vertices))
        np.random.shuffle(temp)
        subsample += temp[:subn]
    subsample = torch.LongTensor(subsample)
    vertices_sub = vertices[subsample.long()]
    targets_sub = targets[subsample.long()]
    edges_sub = adjacency_matrix[vertices[subsample.long()].long(), targets[subsample.long()].long()]
    
    v_i = vertices_sub.long()
    v_j = targets_sub.long()
    sign = (edges_sub-0.5)*2
    
    for repeat in range(num_HMC_step):
        
        e_i = net_embedding(v_i.to(device))
        e_j = net_embedding(v_j.to(device))
        log_likelihood = line_loss(e_i, e_j, sign.to(device))
        tmp = - loss_func(torch.sigmoid(net(net_embedding(vertices_sub.to(device).long()), targets_sub.to(device).long())), edges_sub.to(device)) * num_observed
        
        log_likelihood = 0.01*log_likelihood + tmp

        optim_embedding.zero_grad()
        log_likelihood.backward()
        optim_embedding.step()
        del e_i,e_j
        del log_likelihood,tmp
    
    
    optim_net_new.zero_grad()
    loss = - loss_func(torch.sigmoid(net(net_embedding(vertices_sub.to(device).long()), targets_sub.to(device).long())), edges_sub.to(device)) * num_observed
    loss.backward()
    optim_net_new.step()
    del loss
    
    if epoch%10==0:
        ans = cal_dev()
        sparse_path_new.append(np.array(ans))
       
    if epoch%10==0:
        with torch.no_grad():
        	output = torch.zeros(num_vertices, num_vertices)
            for i in range(num_vertices):
                embed_tmp = net_embedding(torch.Tensor([i]).repeat(num_vertices).to(device).long())
                tmp_output = net(embed_tmp, torch.Tensor(np.arange(num_vertices)).to(device).long())
                output[i,:] = tmp_output.cpu().detach()
                del tmp_output
                del embed_tmp
            prediction = output > 0
            accuracy = prediction.eq(edges.data.byte()).sum().item() / num_observed
            target_true = torch.sum(edges.data == 1).float()
            predicted_true = torch.sum(prediction == 1).float()
            correct_true = torch.sum((prediction == 1) * (edges.data == 1)).float()
            recall = correct_true / target_true
            precision = correct_true / predicted_true
            prob = torch.sigmoid(output)
            auc = roc_auc_score(edges, prob) 
            loss = loss_func(prob, edges)
            auc_path.append(auc)
            recall_path.append(recall)
            precision_path.append(precision)
            del output, prob
            print('epoch: ', epoch, ' loss: ', loss, ' auc: ', auc, ' accuracy: ', accuracy, ' recall:', recall, ' precision: ', precision) 
            print(evaluate_embedding_sup(net_embedding(vertices.unique().to(device).long()).cpu().detach().numpy()))

torch.save(net.state_dict(), 'email_net_lambda_1.pth')
torch.save(net_embedding.state_dict(), 'email_embed_lambda_1.pth')



###############
# convergence #
###############
sparse_path_new_decay = []
auc_path_decay = []
recall_path_decay = []
precision_path_decay = []

optim_embedding = SGHMC(net_embedding.parameters(), lr=3e-02, fraction=10.0, temperature=1e-6) 
optim_net_new = newAdamSGDWeighted(net.parameters(), lr=8e-4, lr_sparse=1.0) 

lambda2 = lambda epoch: 1/((epoch+1)**0.9) 
embed_step_scheduler = LambdaLR(optim_embedding, lambda2)

lambda2 = lambda epoch: 1/((epoch+1)**0.9) 
para_step_scheduler = LambdaLR(optim_net_new, lambda2)

for epoch in range(1,101):    
    subsample = np.arange(num_observed)
    subsample = []
    for i in range(num_vertices):
        temp = list(range(i*num_vertices, (i+1)*num_vertices))
        np.random.shuffle(temp)
        subsample += temp[:subn]
    subsample = torch.LongTensor(subsample)
    vertices_sub = vertices[subsample.long()]
    targets_sub = targets[subsample.long()]
    edges_sub = adjacency_matrix[vertices[subsample.long()].long(), targets[subsample.long()].long()]
    
    v_i = vertices_sub.long()
    v_j = targets_sub.long()
    sign = (edges_sub-0.5)*2
    
    for repeat in range(num_HMC_step):
        
        e_i = net_embedding(v_i.to(device))
        e_j = net_embedding(v_j.to(device))
        log_likelihood = line_loss(e_i, e_j, sign.to(device))
        tmp = - loss_func(torch.sigmoid(net(net_embedding(vertices_sub.to(device).long()), targets_sub.to(device).long())), edges_sub.to(device)) * num_observed
        log_likelihood = 0.01*log_likelihood + tmp

        optim_embedding.zero_grad()
        log_likelihood.backward()
        optim_embedding.step()
        del e_i,e_j
        del log_likelihood,tmp
    embed_step_scheduler.step()
    print('para: ', optim_embedding.state_dict()['param_groups'][0]['lr'])

    optim_net_new.zero_grad()
    loss = - loss_func(torch.sigmoid(net(net_embedding(vertices_sub.to(device).long()), targets_sub.to(device).long())), edges_sub.to(device)) * num_observed
    loss.backward()
    optim_net_new.step()
    del loss
    para_step_scheduler.step()
    print('para: ', optim_net_new.state_dict()['param_groups'][0]['lr'])

    if epoch%10==0:
        ans = cal_dev()
        sparse_path_new_decay.append(np.array(ans))
     
    if epoch%10==0:
        with torch.no_grad():
            output = torch.zeros(num_vertices, num_vertices)
            for i in range(num_vertices):
                embed_tmp = net_embedding(torch.Tensor([i]).repeat(num_vertices).to(device).long())
                tmp_output = net(embed_tmp, torch.Tensor(np.arange(num_vertices)).to(device).long())
                output[i,:] = tmp_output.cpu().detach()
                del tmp_output
                del embed_tmp
            prediction = output > 0
            accuracy = prediction.eq(edges.data.byte()).sum().item() / num_observed
            target_true = torch.sum(edges.data == 1).float()
            predicted_true = torch.sum(prediction == 1).float()
            correct_true = torch.sum((prediction == 1) * (edges.data == 1)).float()
            recall = correct_true / target_true
            precision = correct_true / predicted_true
            prob = torch.sigmoid(output)
            auc = roc_auc_score(edges, prob) 
            loss = loss_func(prob, edges)
            auc_path_decay.append(auc)
            recall_path_decay.append(recall)
            precision_path_decay.append(precision)
            del output, prob
            print('epoch: ', epoch, ' loss: ', loss, ' auc: ', auc, ' accuracy: ', accuracy, ' recall:', recall, ' precision: ', precision) 
            print(evaluate_embedding_sup(net_embedding(vertices.unique().to(device).long()).cpu().detach().numpy()))

torch.save(net.state_dict(), 'email_net_lambda_1_decay.pth')
torch.save(net_embedding.state_dict(), 'email_embed_lambda_1_decay.pth')
np.savetxt('email_path_lambda_1_decay.csv', np.array(sparse_path_new+sparse_path_new_decay))



###########
# results #
###########
def get_precisionK(reconstructed_adj, graph, max_index):
    print("get precisionK...")
    reconstructed_adj = reconstructed_adj.reshape(-1)
    sortedInd = np.argsort(reconstructed_adj)
    cur = 0
    count = 0
    precisionK = []
    sortedInd = sortedInd[::-1]
    gg = nx.adjacency_matrix(graph).todense()
    for ind in sortedInd:
        x = ind // graph.number_of_nodes()
        y = ind % graph.number_of_nodes()
        count += 1
        if gg[x,y] == 1 or x==y:
            cur += 1 
        precisionK.append(1.0 * cur / count)
        if count > max_index:
            break
    return precisionK

#lambda=1.0, threshold=1e-4
#lambda=3.0, threshold=1e-6
#lambda=5.0, threshold=1e-6
#lambda=7.0, threshold=1e-6
cnt=[]
for j in range(z_dim):
    if np.array(sparse_path_new+sparse_path_new_decay)[-1][j]>1e-4:
        cnt.append(j)
print(len(cnt))
reduced = net_embedding(vertices.unique().to(device).long())[:,torch.Tensor(cnt).long()] 
print(evaluate_embedding_sup(reduced.cpu().detach().numpy()))


cnt_no = []
for j in range(z_dim):
    if j not in set(cnt):
        cnt_no.append(j)
print(len(cnt_no))

output = torch.zeros(num_vertices, num_vertices)
for i in range(num_vertices):
    embed_tmp = net_embedding(torch.Tensor([i]).repeat(num_vertices).to(device).long())
    embed_tmp[:, torch.Tensor(cnt_no).long()] = 0.0
    tmp_output = net(embed_tmp, torch.Tensor(np.arange(num_vertices)).to(device).long())
    output[i,:] = tmp_output.cpu().detach()
    del tmp_output
    del embed_tmp
    
output = output.reshape((-1,))
prob = torch.sigmoid(output)
auc = roc_auc_score(edges, prob.numpy()) 

G = nx.read_edgelist('email-Eu-core.txt', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
pre1 = get_precisionK(prob.numpy(), G, 10000)
np.savetxt('email_pre_lambda_1_decay.csv', np.array(pre1))



