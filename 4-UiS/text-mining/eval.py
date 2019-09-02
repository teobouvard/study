

def compute_confusion_matrix(actual, predicted, class_index):
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for act, pred in zip(actual, predicted):
        if act == pred:
            if pred == class_index:
                tp += 1
            else:
                tn += 1
        else:
            if pred == class_index:
                fp += 1
            else:
                fn += 1

    return [tp, tn, fp, fn]


def main(actual, predicted):
    
    metrics = {}

    for class_index in set(actual):
        metrics[str(class_index)] = compute_confusion_matrix(actual, predicted, class_index)


    micro_accuracy = sum(metrics[item][0]+metrics[item][1] for item in metrics) / sum(sum(metric) for metric in metrics.values()) #sum(metrics[item][0]+metrics[item][1] for item in metrics) / sum(metrics.values())
    print(micro_accuracy)

def compute_scores(tp, tn, fp, fn):
    sum_all = tp+tn+fp+fn
    accuracy = (tp+tn)/sum_all
    err_rate = (fp+fn)/sum_all
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1_score = (2*precision*recall)/(precision+recall)
    
    print("Accuracy : ", accuracy)
    print("Error rate : ", err_rate)
    print("Recall : ", recall)
    print("Precision : ", precision)
    print("F1 : ", f1_score)


if __name__ == "__main__":
    
    actual = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
    predicted = [1, 1, 1, 2, 3, 2, 3, 1, 3, 4, 2, 3]

    main(actual, predicted)


# print("Type 1 rate : ", fp/(fp+tn)
# print("Type 2 rate : ", fn/(fn+tp))