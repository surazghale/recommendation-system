import flask
#from flask import json
import pandas as pd
import json
import csv
from sqlalchemy import create_engine
from json import dumps
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
import numpy as np
import myapp
import pickle
from flask import request
app = flask.Flask(__name__)
app.config["DEBUG"] = True

k=10
metric='cosine'
pd.read_pickle('gurkha.pkl')
pd.read_pickle('users.pkl')
un_user = open("users.pkl","rb")
un_user_id = pickle.load(un_user)
books = pd.read_pickle('books.pkl')
un_book = open('books.pkl','rb')
book_unpickle = pickle.load(un_book)
ratings_matrix=pd.read_pickle('ratings.pkl')
pickle_off = open("ratings.pkl","rb")
matrix=pd.read_pickle('matrixex_1.pkl')
pickle_offs = open("matrixex_1.pkl","rb")
emp = pickle.load(pickle_offs)
#A = np.squeeze(np.asarray(emp))
item_id=pd.read_pickle('book_id.pkl')

#book_id=pd.read_pickle('books.pkl')
unpickle_book = open("books.pkl","rb")
un_item_id = pickle.load(unpickle_book)

#=======for top ratings======
top_ratings=pd.read_pickle('top_ratings.pkl')
pickle_un = open("top_ratings.pkl","rb")
top_books = pickle.load(pickle_un)


jpt = 32000
shape = emp.shape[1] -jpt
#print(shape)
#loc = emp.index.get_loc(2110)
#print(loc)
#======================APPEND DATA FOR NEW USERS=============
# users_id = 274069
# books_id = '0000913154'
# emp_t = emp.T
# emp_t[users_id] = 0
# emps = emp_t.T
# new_df = pd.DataFrame({books_id: [9]}, index=[users_id])
# emps.update(new_df)

# jpt = emps.copy(deep = True)
# print(jpt)

#=============APPEND DATA FOR NEW USERS=====================+*********************************&&&&&&&&&&&&&&&&&&&&&&
#users_id = 274066
book_id = '0000913154'
rating = 0
# emps = []
@app.route('/append_new_users/<int:user_id>',methods=['GET'])
def append_new_users(user_id):
    # req_data = request.get_json()
    # user_id = req_data['user_id']
    # book_id = req_data['book_id']
    # rating  = req_data['rating']
    global book_id
    global rating
    global emp
    emp_t = emp.T
    emp_t[user_id] = 0
    emp = emp_t.T
    new_df = pd.DataFrame({book_id: [rating]}, index=[user_id])
    emp.update(new_df)
    
    emp.to_csv('example.csv')
    with open(r'example.csv', 'r',newline='',) as csvFile:
        csv_reader = csv.reader(csvFile)
        with open(r'new_example.csv','w',newline='',) as new_file:
            csv_writer = csv.writer(new_file)
            for line in csv_reader:
                csv_writer.writerow(line)


    emp = pd.read_csv('new_example.csv',sep=',', error_bad_lines=False, encoding="latin-1")
    # emp.read_csv('new_users.csv', sep=',', error_bad_lines=False, encoding="latin-1")
    # print(emp)
    ratings_count = pd.DataFrame(top_books.groupby(['ISBN'])['bookRating'].sum())
    top10 = ratings_count.sort_values('bookRating', ascending = False).head(10)
    #print("Following books are recommended")
    a = top10.merge(book_unpickle, left_index = True, right_on = 'ISBN')

    # print(a)
    # print(top10)
    # return 'jpt'
    # b = top10
    # b = top10.rename_axis(None)
    c = a.drop(columns=['bookRating','bookAuthor','yearOfPublication','publisher'])
    # c =b.drop(['bookRating','bookAuthor','yearOfPublication','publisher'],axis = 1)
    d = c.to_dict('list')
    # print(d)
    # return 'jpt'
    e = list(d.values())
    # print(e)
    f = e[0][0:]
    g = e[1][0:]
    h = dict(zip(f,g))
    # print(h)
    i = json.dumps(h)
    return i
    # return 'jpt'




#================UPDATE DATAFRAME  ===================+++++++++++******************&&&&&&&&&&&&&&&&&&&&
#book_id = '0000913154'
#ratings = 7
#user_id = 4385
@app.route('/update_csv',methods=['POST'])
def update_csv():
    req_data = request.get_json()
    user_id = req_data['user_id']
    book_id = req_data['book_id']
    rating  = req_data['rating']
    #print(user_id)
    #print(user_id)
    #return 'jpt'
    global emp
    new_df = pd.DataFrame({book_id: [rating]}, index=[user_id])
    emp.update(new_df)
    
    print(emp)
    
    return 'jpt'

#==========top RATINGS=========================+++++++++++++++++++++++&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
@app.route('/top_books', methods=['GET'])
def top_rating():
    global emp
    ratings_count = pd.DataFrame(top_books.groupby(['ISBN'])['bookRating'].sum())
    top10 = ratings_count.sort_values('bookRating', ascending = False).head(5)
    print("Following books are recommended")
    a = top10.merge(book_unpickle, left_index = True, right_on = 'ISBN')

    # print(a)
    # print(top10)
    # return 'jpt'
    # b = top10
    # b = top10.rename_axis(None)
    c = a.drop(columns=['bookRating','bookAuthor','yearOfPublication','publisher'])
    # c =b.drop(['bookRating','bookAuthor','yearOfPublication','publisher'],axis = 1)
    d = c.to_dict('list')
    # print(d)
    # return 'jpt'
    e = list(d.values())
    f = e[0][0:]
    g = e[1][0:]
    h = dict(zip(f,g))
    i = json.dumps(h)
    return i

    # f = e[0]
    # # print(f)

    # # print(e)
    # # return json.dumps(e)
    # # d = c.to_string(header=False)
    # # # d = b.drop(index='Index', columns='bookRating')
    # # print(d.split('\n'))
    # # print(e)
    # # return 'jpt'

    # # c = b.to_string(header=False)
    # # return c.to_dict()
    # # c = list(b.values())
    # # return str(c)
    # alist = []
    # for j in range(len(f)):
    #     alist.append(f[j])
    # # #  # print(type(alist))
    # return json.dumps(alist)

#============================================RECOMMENDATIONS==========================================================
@app.route('/<item_id>', methods=['GET'])
def home(item_id):
    return item_id
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"
    #pd.read_pickle('gurkha.pkl')

@app.route('/sim/<item_id>', methods=['GET'])
def findksimilaritems(item_id):
    global emp
    global metric
    global k
    
    similarities=[]
    indices=[]
    ratings=emp.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)
    
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()

    return  similarities,indices
    
    
#with myapp.app.test_request_context('/', method='POST'):
#myapp.findksimilaritems('0001056107',ratings_matrix)


@app.route('/prediction/<int:user_id>/<item_id>', methods=['GET'])
def predict_itembased(user_id,item_id):
    #user_id = 2033
    #item_id = '0001056107'
    # dic = {item_id: }
    prediction= wtd_sum =0
    global emp
    global metric
    global k
    
    similarities=[]
    indices=[]
    locs = emp.index.get_loc(user_id)
    loc = emp.columns.get_loc(item_id)

    similarities, indices=findksimilaritems(item_id)
    sum_wt = np.sum(similarities)-1
    product=1
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == loc:
            continue;
        else:
            product = emp.iloc[locs,indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product                              
    prediction = int(round(wtd_sum/sum_wt))
    
    #in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    #which are handled here as below //code has been validated without the code snippet below, below snippet is to avoid negative
    #predictions which might arise in case of very sparse datasets when using correlation metric
    if prediction <= 0:
        prediction = 1   
    elif prediction >10:
        prediction = 10
    print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))      
    
    return prediction  
    
    
    #return '{},{}'.format(similarities,indices)
    
    #similar users based on correlation coefficients
    #sum_wt = np.sum(similarities)-1
    #Sreturn str(sum_wt)
    #return prediction
    #return str(locs)
    
    

#with myapp.app.test_request_context('/', method='POST'):
#myapp.predict_itembased(11676,'0001056107',ratings_matrix)
# user_id = 4385
@app.route('/reco/<int:user_id>', methods=['GET'])
def recommendItem(user_id): 
    # global user_id
    global emp
    global metric
    global k
    global jpt
    #emps = emp.T
    prediction = []
    for i in range(emp.shape[1] - jpt):
        #if((emp.iloc[user_id][i]) == 0):
        if (emp[str(emp.columns[i])][user_id] !=0): #not rated already
            prediction.append(predict_itembased(user_id, str(emp.columns[i])))
        else:                    
            prediction.append(-1) #for already rated items
    
    prediction = pd.Series(prediction)
    prediction = prediction.sort_values(ascending=False)
    recommended = prediction[:5]
    #val = books.bookTitle[recommended.index[:10]]
    #return str(val)
    #return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"
    #pd.read_pickle('gurkha.pkl')
   #print("As per {0} approach....Following books are recommended...".format(select.value))
                
    #for i in range(len(recommended)):
        #result = print("{0}. {1}".format(i+1,books.bookTitle[recommended.index[i]].encode('utf-8')))                 
        #return result.to_dict()          
    #predictions = sorted(prediction,reverse=True)

    
    #recommended = predictions[:10]
    #val = books.bookTitle[recommended.index[:10]].encode('utf-8')
    #return str(val)
    
    dictOfWords = { books.ISBN[recommended.index[i]] : books.bookTitle[recommended.index[i]] for i in range(0, len(recommended) ) }
    #dictOfWords.values()
    i = json.dumps(dictOfWords)
    return i
    #return str(dictOfWords)
    #return json.dumps(dictOfWords.values(), ensure_ascii=False)
    # print(dictOfWords)
    # return 'jpt'
    # a = list(dictOfWords.values())
    # alist = []
    # for j in range(len(a)):
    #     alist.append(a[j].decode('utf-8'))
    # return json.dumps(alist)
    #return str(np.array(dictOfWords.values()))
    # dict_list=[]
    # for i,j in dictOfWords.items():
    #     dict_list.append ((i,j))
    # return dict_list
                        

app.run()