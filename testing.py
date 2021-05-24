from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from flask_ngrok import run_with_ngrok
from flask import Flask, Response, request, jsonify, render_template

driver = webdriver.Chrome('chromedriver',options=chrome_options)
app = Flask(__name__,template_folder='template')
run_with_ngrok(app)

@app.route('/')
def index():
    return render_template('testing.html')

@app.route('/testing2', methods=['POST'])
def testing():
    first_name = request.form['fname']
    if first_name == 'yes' or first_name == 'Yes' or first_name == 'YES':
        return render_template('testing2.html')
@app.route('/testing3', methods=['POST'])
def testing1():
    amazon_link = request.form['amazon_link']
    # train Data
    trainData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv")
    # test Data
    testData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/test.csv")
    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df = 5,
                                max_df = 0.8,
                                sublinear_tf = True,
                                use_idf = True)
    train_vectors = vectorizer.fit_transform(trainData['Content'])
    test_vectors = vectorizer.transform(testData['Content'])
    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, trainData['Label'])
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1
    # results
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    report = classification_report(testData['Label'], prediction_linear, output_dict=True)
    print('positive: ', report['pos'])
    print('negative: ', report['neg'])

    reviewlist = []
    def get_url(search_term):
      template = "{}"
      return template.format(search_term)
    product = amazon_link
    #print(product)          #product search
    url = get_url(product)  #Product link
    driver.get(url)
    """Extract the collection"""
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    sub_review_url = soup.find('a', {'data-hook': 'see-all-reviews-link-foot'})
    review_url = sub_review_url.get('href')
    driver.get("https://www.amazon.in"+review_url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    reviews = soup.find_all('div', {'data-hook': 'review'})
    for item in reviews:
        product_name_ = soup.title.text.replace('Amazon.in:Customer reviews:','').strip()  
        review = [
        #'product': soup.title.text.replace('Amazon.in:Customer reviews:','').strip(),
        #'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
        #'rating': float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
        item.find('span', {'data-hook': 'review-body'}).text.strip(),
        ]
        #print(review)
        reviewlist.append(review)
    def view_comments():
        reviews = soup.find_all('div', {'data-hook': 'review'})
        for item in reviews:
            review = [
            #'product': soup.title.text.replace('Amazon.in:Customer reviews:','').strip(),
            #'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
            #'rating': float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
            item.find('span', {'data-hook': 'review-body'}).text.strip(),
            ]
            #print(review)
            reviewlist.append(review)
    for x in range(1,30):
        next_page = soup.find('div', {'class': 'a-form-actions a-spacing-top-extra-large'})
        next_page1 = next_page.find('li', {'class': 'a-last'})
        next_page2 = next_page1.find('a')
        next_page3 = next_page2.get('href')
        driver.get("https://www.amazon.in"+next_page3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        view_comments()
        if not soup.find('li', {'class':'a-disabled a-last'}):
            pass
        else:
            break
    #print(*reviewlist, sep = "\n")
    df = pd.DataFrame(reviewlist)
    df.to_excel('livedataset.xlsx', index=False)
    print('Finished..')
    count_pos = 0
    count_neg = 0
    for i in range(len(reviewlist)):
        read_review = str(reviewlist[i])
        review_vector = vectorizer.transform([read_review]) # vectorizing
        x = (classifier_linear.predict(review_vector))
        if x == 'pos':
            count_pos = count_pos + 1
        elif x == 'neg':
            count_neg = count_neg + 1
    positive_reviews = float ((count_pos / len(reviewlist)) * 100)
    negative_reviews = float ((count_neg / len(reviewlist))*100)
    print("According to the Support Vector Machine(SVM) approach the product sentiment is %.2f Positive" % positive_reviews)
    df=pd.read_csv("https://raw.githubusercontent.com/ROHITSALUNKE1998/testing/main/test.txt",sep='\t',names=['like','txt'])
    df.head()
    nltk.download('stopwords')
    stopset=set(stopwords.words('english'))
    vectorizer=TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii',stop_words=stopset)
    y=df.like
    x_nb_=vectorizer.fit_transform(df.txt)
    x_train, x_test, y_train, y_test= train_test_split(x_nb_,y,random_state=0)
    clf = naive_bayes.MultinomialNB()
    clf.fit(x_train, y_train)
    roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])

    count_pos_nb = 0
    count_neg_nb = 0
    for j in range(len(reviewlist)):
        review_nb = np.array([str(reviewlist[j])])
        review_vector_nb = vectorizer.transform(review_nb)
        x_nb = (clf.predict(review_vector_nb))
        if x_nb == [1]:
            count_pos_nb = count_pos_nb + 1
        elif x_nb == [0]:
            count_neg_nb = count_neg_nb + 1
    positive_reviews_nb = float ((count_pos_nb / len(reviewlist)) * 100)
    negative_reviews_nb = float ((count_neg_nb / len(reviewlist))*100)
    print("According to the Naive Bayes approach the product sentiment is %.2f Positive" % positive_reviews_nb)
    positive_reviews_all = float ((positive_reviews+positive_reviews_nb)/2)
    print("The overall Sentiment of "+product_name_+" is %.2f Positve" % positive_reviews_all)
    
    return '<h1 style="color:darkblue; font-family: cursive;">The overall Sentiment of '+product_name_+' is %.2f positive </h1> <br/><a href="/">Back Home</a>'% (positive_reviews_all)



if __name__ == '__main__':
    app.run()
