

def compute_confusion_matrix(actual, predicted):
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(actual)):
        if predicted[i] == actual[i]:
            if actual[i]:
                tp += 1
            else:
                tn += 1
        else:
            if predicted[i]:
                fp += 1
            else:
                fn += 1

    return tp, tn, fp, fn


def main(actual, predicted):
    
    tp, tn, fp, fn = compute_confusion_matrix(actual, predicted)
    sum_all = tp+tn+fp+fn
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)

    print("Accuracy : ", (tp+tn)/sum_all)
    print("Error rate : ", (fp+fn)/sum_all)
    print("Recall : ", recall)
    print("Precision : ", precision)
    print("F1 : ", (2*precision*recall)/(precision+recall))
    print("Type 1 rate : ", fp/(fp+tn))
    print("Type 2 rate : ", fn/(fn+tp))



if __name__ == "__main__":
    
    actual = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    predicted = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]

    main(actual, predicted)
