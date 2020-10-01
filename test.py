import torch
from utils import l2_normalize
import numpy as np
import os


def get_normal_vector(model, train_normal_loader_for_test, cal_vec_batch_size, feature_dim, use_cuda):
    total_batch = int(len(train_normal_loader_for_test))
    print("=====================================Calculating Average Normal Vector=====================================")
    if use_cuda:
        normal_vec = torch.zeros((1, 512)).cuda()
    else:
        normal_vec = torch.zeros((1, 512))
    for batch, (normal_data, idx) in enumerate(train_normal_loader_for_test):
        if use_cuda:
            normal_data = normal_data.cuda()
        _, outputs = model(normal_data)
        outputs = outputs.detach()
        normal_vec = (torch.sum(outputs, dim=0) + normal_vec * batch * cal_vec_batch_size) / (
                (batch + 1) * cal_vec_batch_size)
        print(f'Calculating Average Normal Vector: Batch {batch + 1} / {total_batch}')
    normal_vec = l2_normalize(normal_vec)
    return normal_vec


def split_acc_diff_threshold(model, normal_vec, test_loader, use_cuda):
    """
    Search the threshold that split the scores the best and calculate the corresponding accuracy
    """
    total_batch = int(len(test_loader))
    print("================================================Evaluating================================================")
    total_n = 0
    total_a = 0
    threshold = np.arange(0., 1., 0.01)
    total_correct_a = np.zeros(threshold.shape[0])
    total_correct_n = np.zeros(threshold.shape[0])
    for batch, batch_data in enumerate(test_loader):
        if use_cuda:
            batch_data[0] = batch_data[0].cuda()
            batch_data[1] = batch_data[1].cuda()
        n_num = torch.sum(batch_data[1]).cpu().detach().numpy()
        total_n += n_num
        total_a += (batch_data[0].size(0) - n_num)
        _, outputs = model(batch_data[0])
        outputs = outputs.detach()
        similarity = torch.mm(outputs, normal_vec.t())
        for i in range(len(threshold)):
            prediction = similarity >= threshold[i]  # If similarity between sample and average normal vector is smaller than threshold, then this sample is predicted as anormal driving which is set to 0
            correct = prediction.squeeze() == batch_data[1]
            total_correct_a[i] += torch.sum(correct[~batch_data[1].bool()])
            total_correct_n[i] += torch.sum(correct[batch_data[1].bool()])
        print(f'Evaluating: Batch {batch + 1} / {total_batch}')
        print('\n')
    acc_n = [(correct_n / total_n) for correct_n in total_correct_n]
    acc_a = [(correct_a / total_a) for correct_a in total_correct_a]
    acc = [((total_correct_n[i] + total_correct_a[i]) / (total_n + total_a)) for i in range(len(threshold))]
    best_acc = np.max(acc)
    idx = np.argmax(acc)
    best_threshold = idx * 0.01
    return best_acc, best_threshold, acc_n[idx], acc_a[idx], acc, acc_n, acc_a


def cal_score(model_front_d, model_front_ir, model_top_d, model_top_ir, normal_vec_front_d, normal_vec_front_ir,
              normal_vec_top_d, normal_vec_top_ir, test_loader_front_d, test_loader_front_ir, test_loader_top_d,
              test_loader_top_ir, score_folder, use_cuda):
    """
    Generate and save scores of top_depth/top_ir/front_d/front_ir views
    """
    assert int(len(test_loader_front_d)) == int(len(test_loader_front_ir)) == int(len(test_loader_top_d)) == int(
        len(test_loader_top_ir))
    total_batch = int(len(test_loader_front_d))
    sim_list = torch.zeros(0)
    sim_1_list = torch.zeros(0)
    sim_2_list = torch.zeros(0)
    sim_3_list = torch.zeros(0)
    sim_4_list = torch.zeros(0)
    label_list = torch.zeros(0).type(torch.LongTensor)
    for batch, (data1, data2, data3, data4) in enumerate(
            zip(test_loader_front_d, test_loader_front_ir, test_loader_top_d, test_loader_top_ir)):
        if use_cuda:
            data1[0] = data1[0].cuda()
            data1[1] = data1[1].cuda()
            data2[0] = data2[0].cuda()
            data2[1] = data2[1].cuda()
            data3[0] = data3[0].cuda()
            data3[1] = data3[1].cuda()
            data4[0] = data4[0].cuda()
            data4[1] = data4[1].cuda()

        assert torch.sum(data1[1] == data2[1]) == torch.sum(data2[1] == data3[1]) == torch.sum(data3[1] == data4[1]) == \
               data1[1].size(0)

        out_1 = model_front_d(data1[0])[1].detach()
        out_2 = model_front_ir(data2[0])[1].detach()
        out_3 = model_top_d(data3[0])[1].detach()
        out_4 = model_top_ir(data4[0])[1].detach()

        sim_1 = torch.mm(out_1, normal_vec_front_d.t())
        sim_2 = torch.mm(out_2, normal_vec_front_ir.t())
        sim_3 = torch.mm(out_3, normal_vec_top_d.t())
        sim_4 = torch.mm(out_4, normal_vec_top_ir.t())
        sim = (sim_1 + sim_2 + sim_3 + sim_4) / 4

        sim_list = torch.cat((sim_list, sim.squeeze().cpu()))
        label_list = torch.cat((label_list, data1[1].squeeze().cpu()))
        sim_1_list = torch.cat((sim_1_list, sim_1.squeeze().cpu()))
        sim_2_list = torch.cat((sim_2_list, sim_2.squeeze().cpu()))
        sim_3_list = torch.cat((sim_3_list, sim_3.squeeze().cpu()))
        sim_4_list = torch.cat((sim_4_list, sim_4.squeeze().cpu()))
        print(f'Evaluating: Batch {batch + 1} / {total_batch}')

    np.save(os.path.join(score_folder, 'score_front_d.npy'), sim_1_list.numpy())
    print('score_front_d.npy is saved')
    np.save(os.path.join(score_folder, 'score_front_IR.npy'), sim_2_list.numpy())
    print('score_front_IR.npy is saved')
    np.save(os.path.join(score_folder, 'score_top_d.npy'), sim_3_list.numpy())
    print('score_top_d.npy is saved')
    np.save(os.path.join(score_folder, 'score_top_IR.npy'), sim_4_list.numpy())
    print('score_top_IR.npy is saved')

