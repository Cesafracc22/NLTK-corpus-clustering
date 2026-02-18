from src.preprocessing.nltk_preprocessing import NLTKPreprocessingPipeline
import argparse


def main():
    parser = argparse.ArgumentParser(description='NLTK preprocessing pipeline')
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--remove_stopwords", action="store_true")
    parser.add_argument("--stem", action="store_true")
    parser.add_argument("--tfidf", action="store_true")
    parser.add_argument("--word2vec", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    pipeline = NLTKPreprocessingPipeline(args)

    # Fit and transform training data
    result = pipeline.fit_transform()
    print(f"Training result type: {type(result)}")
    if hasattr(result, 'shape'):
        print(f"Training result shape: {result.shape}")
    else:
        print(f"Training result length: {len(result)}")

    # If split, also transform test data
    if args.split:
        train_data, test_data = pipeline.get_train_test_split()
        print(f"\nTrain size: {len(train_data)}")
        print(f"Test size: {len(test_data)}")

        test_result = pipeline.transform()
        print(f"\nTest result type: {type(test_result)}")
        if hasattr(test_result, 'shape'):
            print(f"Test result shape: {test_result.shape}")
        else:
            print(f"Test result length: {len(test_result)}")


if __name__ == "__main__":
    main()
