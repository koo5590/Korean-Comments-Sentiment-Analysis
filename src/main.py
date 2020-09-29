import argparse

# model 1 ~ 3
from Train import Trainer
from Test import Tester
from interactive_predict import interactive_predict


# model 4
from model_4.first_Train import TrainerFirst
from model_4.second_classifier import SecondClassifier
from model_4.third_Train import TrainerThird
from model_4.predict import Predictor

# new_model 4
from new_model_4.first_Train import FirstTrainer
from new_model_4.second_classifier import SecondClassifier_n
from new_model_4.mmd_trainer import Trainermmd
from new_model_4.predict import Predictor_n


from utils import init_logger, set_seeds, load_tokenizer

def main(args):
    init_logger()
    set_seeds()
    tokenizer = load_tokenizer(args)

    if args.do_train:
        if args.model_4:
            first =TrainerFirst(args, tokenizer)
            first.train()
            second = SecondClassifier(args, tokenizer)
            second.classifier()
            third = Trainermmd(args, tokenizer)
            third.train()
        elif args.new_model_4:
            first =FirstTrainer(args, tokenizer)
            first.train()
            second = SecondClassifier_n(args, tokenizer)
            second.classifier()
            third = Trainermmd_n(args, tokenizer)
            third.train()
        else:
            trainer = Trainer(args, tokenizer)
            trainer.train()
    elif args.do_test:
        if args.model_4 or args.new_model_4:
            tester = Predictor(args, tokenizer)
            tester.predict()
        else:    
            tester = Tester(args, tokenizer)
            tester.test()
    elif args.do_interactive:
        interactive_predict(args)

if __name__ == '__main__':
    # model_name_or_path
    # kc-bert : beomi/kcbert-base
    # ko-electra : monologg/koelectra-base-v2-discriminator
    # bert-m : bert-base-multilingual-cased

    # test_data_dir (100 lines)
    # tv: ../data/finetuning/tv
    # sports: ../data/finetuning/sports

    #kcbert_lstm_sports/model_sports.pt
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_model_dir", default="./models/new_model_4", type=str, help="Path load testing model")
    parser.add_argument("--save_model_dir", default="./models", type=str, help="Path to save trained model")

    parser.add_argument("--data_dir", default="./data", type=str, help="Train file")
    parser.add_argument("--train_data_dir", default="movie_train", type=str, help="Train file")
    parser.add_argument("--test_data_dir", default="sports_test", type=str, help="Test file")
    parser.add_argument("--test_output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--bert_type", default="beomi/kcbert-base", type=str)

    # model 4
    parser.add_argument("--first_domain_classifier_output", default="./models/new_model_4/first", type=str, help="Output file for")
    parser.add_argument("--scond_classifier_output", default="./models/new_model_4/second", type=str, help="Output file for")
    parser.add_argument("--third_sentiment_classifier_output", default="./models/new_model_4", type=str, help="Output file for")
    parser.add_argument("--target_data", default="sports", type=str, help="Which data for the target domain")
    parser.add_argument("--model4_train_dir", default="sports", type=str, help="Target data to train model 4")

    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")

    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=60, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=180, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=180, help="Save every X updates steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="")
    parser.add_argument('--max_steps', type=int, default=5000, help="Max steps to train")
    parser.add_argument('--early_cnt', type=int, default=3, help="Max steps to train")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="Max sequence length.")
    
    parser.add_argument("--second_finetuning", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_interactive", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--model_1", action="store_true", help="Run model_1")
    parser.add_argument("--model_2", action="store_true", help="Run model_2")
    parser.add_argument("--freeze", action="store_true", help="Whether to freeze when training model 2")
    parser.add_argument("--model_3", action="store_true", help="Run model_3")
    parser.add_argument("--model_4", action="store_true", help="Run model_4")
    parser.add_argument("--new_model_4", action="store_true", help="Run new_model_4")

    parser.add_argument("--do_first", action="store_true", help="Train a domain classifier for the model 4")
    parser.add_argument("--do_second", action="store_true", help="Select the most target similar movie data for the model 4")
    parser.add_argument("--do_third", action="store_true", help="Train a sentiment classifier for the model 4")

    args = parser.parse_args()

    main(args)

