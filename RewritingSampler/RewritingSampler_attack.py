from fibber.datasets import get_dataset
from fibber.fibber import Fibber
from fibber.resources import download_all


# The following parameters are copied directly from Fibber's Google Colab tutorial 
# (https://colab.research.google.com/drive/1zefsU19P3HdrBUqJy7HU9b9cSaB_nBMP#scrollTo=uNcmhgzHJ3VQ)
# as they are well-tuned hyperparameters:
wpe_weight = 1000  #@param {type:"raw"}
wpe_threshold = 1. #@param {type:"raw"}
use_threshold = 0.98 #@param {type:"raw"}
use_weight = 2000  #@param {type:"raw"}
gpt2_weight = 10   #@param {type:"raw"}
clf_weight = 3   #@param {type:"raw"}

def download_resources():
    """
    Download all the datasets and resources needed to run this file.
    """
    download_all()

def create_fibber(dataset="ag", block_size = 3):
    """
    Create a Fibber object for the given dataset.
    """
    # The following dictionary is copied directly from Fibber's Google Colab tutorial 
    # (https://colab.research.google.com/drive/1zefsU19P3HdrBUqJy7HU9b9cSaB_nBMP#scrollTo=uNcmhgzHJ3VQ)
    # args starting with "bs_" are hyperparameters for the BertSamplingStrategy.
    arg_dict = {
        "use_gpu_id": 0,
        "gpt2_gpu_id": 0,
        "bert_gpu_id": 0,
        "strategy_gpu_id": 0,

        "bs_block_size": block_size,
        "bs_wpe_weight": wpe_weight,  
        "bs_wpe_threshold": wpe_threshold,
        "bs_use_weight": use_weight,
        "bs_use_threshold": use_threshold,
        "bs_gpt2_weight": gpt2_weight,
        "bs_clf_weight": clf_weight
    }

    # create a fibber object.
    fibber = Fibber(arg_dict, dataset_name=dataset, strategy_name="BertSamplingStrategy", output_dir=".")
    #fibber = Fibber(arg_dict, dataset_name=dataset, strategy_name="TextAttackStrategy", output_dir=".")
    return fibber

def generate_rewritten_testset(fibber, testset, n=1):
    """
    Use the given Fibber object to generate n rewritten sentences for each sentence in the testset.
    """
    rewritten_set = dict()
    rewritten_list = []
    metric_results = []
    for test_instance in testset['data']:
        _, paraphrases, metrics = fibber.paraphrase(test_instance, field_name="text0", n=n)
        metric_results.extend(metrics)
        for paraphrase in paraphrases:
            rewritten_inst = dict()
            rewritten_inst['text0'] = paraphrase
            rewritten_inst['label'] = test_instance['label']
            rewritten_list.append(rewritten_inst)
    rewritten_set['data'] = rewritten_list
    return rewritten_set, metric_results



def calc_accuracy(fibber, testset):
    """
    Calculate accuracy on the given testset.
    """
    tot_num = len(testset['data'])
    acc_pred = 0
    for i in range(tot_num):
        sent = testset['data'][i]['text0']
        y_true = testset['data'][i]['label']
        y_pred = fibber.get_metric_bundle().get_target_classifier().measure_example(None, sent)
        if y_true == y_pred:
            acc_pred += 1
    return acc_pred/tot_num

def main():
    dataset_name = "mnli"
    fibber = create_fibber(dataset_name)
    trainset, testset = get_dataset(dataset_name)
    rewritten_testset, _ = generate_rewritten_testset(fibber, testset)
    before_attack_acc = calc_accuracy(fibber, testset)
    after_attack_acc = calc_accuracy(fibber, rewritten_testset)
    print(before_attack_acc)
    print(after_attack_acc)
    print("Before attack, the classifier trained on dataset %s has an accuracy of %.2f"%(dataset_name, before_attack_acc))
    print("After attack, the classifier trained on dataset %s has an accuracy of %.2f"%(dataset_name, after_attack_acc))

if __name__ == "__main__":
    main()