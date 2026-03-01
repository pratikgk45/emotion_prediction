# Real Time Emotion Prediction
A deep CNN to classify the facial emotion into 7 categories. The model is trained on the **FER-2013** dataset which was a part of kaggle FER-2013 challenge. This dataset consists of 35887 grayscale, 48x48 sized face images with **7 categories** as follows : angry, disgusted, fearful, happy, neutral, sad, surprised. We have used haar cascade method for face detection. For each face in a frame, we found softmax scores for all 7 emotions and emotion with maximum score is displayed.

## Dependencies
To install all required dependencies, run `pip install -r requirements.txt`

## Usage

### Local Development
* Clone the repository and enter into the directory

```bash
git clone https://github.com/pratikgk45/emotion_prediction.git
cd emotion_prediction
```

* Install dependencies

```bash
pip install -r requirements.txt
```

* To train the model, download dataset from [here](https://drive.google.com/file/d/1rkC29dRCaq8TZBh0ZRFANHwW3-PeIsrS/view?usp=sharing) and unzip it to create directory `data` right inside main repository. For training, run

```bash
python train_model.py
```
This will train a new model and its weights will be stored in `model.h5` file. If you want to skip this step, you can use the pre-trained `model.h5` model.

* Run flask app locally

```bash
python main.py
```
Now, go to http://127.0.0.1:5000/ in your browser, give permission to the browser to use webcamera(if asked) and see the predictions.

### AWS Deployment
The application is deployed on AWS using CDK with ECS Fargate.

* Install CDK dependencies

```bash
cd infrastructure
npm install
```

* Deploy to AWS

```bash
npm run cdk bootstrap  # First time only
npm run cdk deploy
```

The deployment creates:
- VPC with public/private subnets
- ECS Fargate cluster
- Application Load Balancer
- Container running the Flask app

Access the application via the Load Balancer DNS output after deployment.