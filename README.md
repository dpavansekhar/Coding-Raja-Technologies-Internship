# Coding-Raja-Technologies-Internship

My 1st Summer Internship at Coding Raja Technologies.

## Tasks Overview

### Task 1: Sentiment Analysis on Social Media Data

For this task, we performed sentiment analysis on social media data using the Sentiment140 dataset. The dataset contains a collection of tweets with pre-labeled sentiments. The main steps involved in this task were:

The Dataset consists of around 1.6 Million tweets . I have worked on to preprocess the dataset and to remove the duplicates from it.

1. **Data Preprocessing**: 
    - Cleaning the text data by removing special characters, numbers, and unnecessary whitespaces.
    - Tokenizing the text and converting it into a format suitable for machine learning models.
    - Splitting the dataset into training and testing sets.

2. **Model Building**:
    - Using Machine Learning Algorithms I have trained a model for sentiment classification.
    - Experimenting with various architectures and hyperparameters to optimize the model's performance.

3. **Evaluation**:
    - Assessing the model's accuracy and other performance metrics on the test dataset.
    - Visualizing the results to understand the model's strengths and weaknesses.

4. **Deployment**:
   - I have created an simple web interface . So that everyone can access it  through the web application.
   - To run the web application after unwrapping the zip file you have downloaded
   - Run Command "python app.py" in the command prompt.
  
   - I have create an 2nd interface too . To access that interface I have created an streamlit_app.py file .
   - In this file I have written my code.
   - Streamlit is an python library which will help the people to develop an simple web interface using the python.
   - To run the file open command prompt and type "streamlit run streamlit_app.py"
     ***Note*** : If you have not installed streamlit library in your desktop run command "pip install streamlit".

     I will share the snapshots of my web interfaces below . Go and check out from there .

     ![Web Interface Developed using HTML,CSS]()
     
     ![Web Interface Developed using Streamlit]()

### Task 2: Image Classification on Food Recognition

In this task, we worked on image classification for food recognition using the Food 101 dataset. The main steps involved in this task were:

The Dataset consists of 101 varities of food items and around 100001 images. I will provide the dataset link below . Pls go and check from there
The variety of food items are

**Classes & Calories**

-The calories per gram for each food item:

- Apple Pie: ~2.5 calories per gram
- Baby Back Ribs: ~3.5 calories per gram
- Baklava: ~5 calories per gram
- Beef Carpaccio: ~2 calories per gram
- Beef Tartare: ~2.5 calories per gram
- Beet Salad: ~0.5 calories per gram
- Beignets: ~3.5 calories per gram
- Bibimbap: ~1.5 calories per gram
- Bread Pudding: ~2.5 calories per gram
- Breakfast Burrito: ~2 calories per gram
- Bruschetta: ~1 calorie per gram
- Caesar Salad: ~0.5 calories per gram
- Cannoli: ~3.5 calories per gram
- Caprese Salad: ~1 calorie per gram
- Carrot Cake: ~3.5 calories per gram
- Ceviche: ~0.5 calories per gram
- Cheese Plate: ~3.5 calories per gram
- Cheesecake: ~3.5 calories per gram
- Chicken Curry: ~1.5 calories per gram
- Chicken Quesadilla: ~2.5 calories per gram
- Chicken Wings: ~3 calories per gram
- Chocolate Cake: ~4 calories per gram
- Chocolate Mousse: ~3 calories per gram
- Churros: ~4 calories per gram
- Clam Chowder: ~1.5 calories per gram
- Club Sandwich: ~2.5 calories per gram
- Crab Cakes: ~2 calories per gram
- Creme Brulee: ~3.5 calories per gram
- Croque Madame: ~3 calories per gram
- Cupcakes: ~3.5 calories per gram
- Deviled Eggs: ~1 calorie per gram
- Donuts: ~4 calories per gram
- Dumplings: ~2.5 calories per gram
- Edamame: ~1 calorie per gram
- Eggs Benedict: ~2.5 calories per gram
- Escargots: ~1 calorie per gram
- Falafel: ~2 calories per gram
- Filet Mignon: ~2.5 calories per gram
- Fish and Chips: ~2.5 calories per gram
- Foie Gras: ~4.5 calories per gram
- French Fries: ~3.5 calories per gram
- French Onion Soup: ~1 calorie per gram
- French Toast: ~2 calories per gram
- Fried Calamari: ~2.5 calories per gram
- Fried Rice: ~1.5 calories per gram
- Frozen Yogurt: ~1 calorie per gram
- Garlic Bread: ~4 calories per gram
- Gnocchi: ~1.5 calories per gram
- Greek Salad: ~0.5 calories per gram
- Grilled Cheese Sandwich: ~3 calories per gram
- Grilled Salmon: ~2 calories per gram
- Guacamole: ~2 calories per gram
- Gyoza: ~2 calories per gram
- Hamburger: ~3.5 calories per gram
- Hot and Sour Soup: ~0.5 calories per gram
- Hot Dog: ~3.5 calories per gram
- Huevos Rancheros: ~2 calories per gram
- Hummus: ~1.5 calories per gram
- Ice Cream: ~2 calories per gram
- Lasagna: ~1.5 calories per gram
- Lobster Bisque: ~1 calorie per gram
- Lobster Roll Sandwich: ~2.5 calories per gram
- Macaroni and Cheese: ~3 calories per gram
- Macarons: ~4 calories per gram
- Miso Soup: ~0.5 calories per gram
- Mussels: ~0.5 calories per gram
- Nachos: ~2.5 calories per gram
- Omelette: ~1.5 calories per gram
- Onion Rings: ~2.5 calories per gram
- Oysters: ~0.5 calories per gram
- Pad Thai: ~2 calories per gram
- Paella: ~1.5 calories per gram
- Pancakes: ~2 calories per gram
- Panna Cotta: ~3.5 calories per gram
- Peking Duck: ~4 calories per gram
- Pho: ~1 calorie per gram
- Pizza: ~2.5 calories per gram
- Pork Chop: ~2.5 calories per gram
- Poutine: ~2.5 calories per gram
- Prime Rib: ~2.5 calories per gram
- Pulled Pork Sandwich: ~2.5 calories per gram
- Ramen: ~1 calorie per gram
- Ravioli: ~1.5 calories per gram
- Red Velvet Cake: ~4 calories per gram
- Risotto: ~1.5 calories per gram
- Samosa: ~2 calories per gram
- Sashimi: ~1 calorie per gram
- Scallops: ~1 calorie per gram
- Seaweed Salad: ~0.5 calories per gram
- Shrimp and Grits: ~2 calories per gram
- Spaghetti Bolognese: ~1.5 calories per gram
- Spaghetti Carbonara: ~2 calories per gram
- Spring Rolls: ~1.5 calories per gram
- Steak: ~2.5 calories per gram
- Strawberry Shortcake: ~3.5 calories per gram
- Sushi: ~1 calorie per gram
- Tacos: ~2 calories per gram
- Takoyaki: ~2.5 calories per gram
- Tiramisu: ~3 calories per gram
- Tuna Tartare: ~1.5 calories per gram
- Waffles: ~2 calories per gram

These values are approximations and can vary based on factors such as ingredients and cooking methods.

---

1. **Data Preparation**:
    - Downloading and organizing the Food 101 dataset.
    - Preprocessing the images by resizing, normalizing, and augmenting them to improve model generalization.

2. **Model Building**:
    - Utilizing the InceptionV3 model from Keras applications as the base model for feature extraction.
    - Adding custom layers on top of InceptionV3 for the specific task of food recognition.
    - Training the model using the preprocessed images and optimizing it using techniques like data augmentation and learning rate scheduling.

3. **Evaluation**:
    - Evaluating the model's performance on a separate validation set.
    - Fine-tuning the model based on the evaluation results to achieve better accuracy and robustness.

I will provide the models I have used through an google drive link . You can download those from there.

**LINKS REQUIRED**

[Dataset of Sentiment Analysis on Social Media Data](https://www.kaggle.com/datasets/kazanova/sentiment140)

[Dataset of Image Classification on Food Recognition](https://www.kaggle.com/datasets/kmader/food41)

[Drive link for the models used in Task 2](https://drive.google.com/drive/folders/1xOSoJaxcs_X_MCMYpmGo2p--MG0M-_nA?usp=sharing)

## Conclusion

This internship provided a great opportunity to apply machine learning techniques to real-world problems. The hands-on experience with sentiment analysis and image classification has been invaluable in enhancing my skills in data science and deep learning.

