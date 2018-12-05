python main.py optim --content-image images/content/columbia.jpg --style-image images/9styles/candy.jpg --output-image images/outputs/test.jpg --alpha 0.1

python main.py eval --content-image images/content/venice-boat.jpg --style-image images/21styles/candy.jpg --model models/21styles.model --content-size 1024  --output-image images/outputs/test.jpg