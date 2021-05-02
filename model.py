
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import build_feedback_matrix, calculate_loss

class CF_Model(object):
    def __init__(self,users,movies):
        self.embedding_vars = {}
        self.train_loss = []
        self.val_loss = []
        self.list_userId = users
        self.list_movieId = movies
        self.num_users = users.shape[0]
        self.num_movies = movies.shape[0]
        self.num_features = 0
        self.models = []

    def embeddings(self):
        return self.embedding_vars
    
    def build_model(self,num_features=30,mu=0,std=1):
        """
            m: number of users
            n: number of movies
            k: number of features or latent factors   
            P: m x k: associative matrix btw Users and Features 
            V: n x k: associative matrix btw Movies and Features
        """
        self.num_features = num_features
        P = np.random.normal(loc=mu,scale=std,size=(self.num_users,num_features))
        V = np.random.normal(loc=mu,scale=std,size=(self.num_movies,num_features))
        self.embedding_vars['P'] = P
        self.embedding_vars['V'] = V
    
    def train(self,train_data,val_data,num_iter=100,learning_rate=0.00002,regulator=0.02):
        train_matrix = build_feedback_matrix(train_data,self.num_users,self.num_movies)
        val_matrix = build_feedback_matrix(val_data,self.num_users,self.num_movies)
        P = self.embedding_vars['P']
        V = self.embedding_vars['V']
        for epoch in range(num_iter):
            
            E = train_matrix - (P @ V.T)
            gradient_P = 2 * (E @ V) - regulator * P
            gradient_V = 2 * (E.T @ P) - regulator * V
            P_new = P + learning_rate * gradient_P
            V_new = V + learning_rate * gradient_V
            P = P_new
            V = V_new

            train_error = calculate_loss(train_matrix,P,V)
            val_error = calculate_loss(val_matrix,P,V)
            self.train_loss.append(train_error)
            self.val_loss.append(val_error)
            
            #print("i: " + str(epoch) + " Train Loss: " + str(round(train_error,4)) + " - Validation Loss: " + str(round(val_error,4)))
            
            embedding_vars = {'P':P, 'V': V}
            self.models.append(embedding_vars)

    def update_best_model(self):
        min_validation_idx = np.argmin(self.val_loss)
        min_validation = np.min(self.val_loss)
        best_model = self.models[min_validation_idx]

        self.embedding_vars['P'] = best_model['P']
        self.embedding_vars['V'] = best_model['V']
        return min_validation
    
    def calculate_test_loss(self,test_data):
        test_matrix = build_feedback_matrix(test_data,self.num_users,self.num_movies)
        P = self.embedding_vars['P']
        V = self.embedding_vars['V']
        test_error = calculate_loss(test_matrix,P,V)
        return test_error

    def calculate_norm(self):
        P = self.embedding_vars['P']
        V = self.embedding_vars['V']
        F = P @ V.T
        norm = np.linalg.norm(F,ord='fro')
        return norm

    def plot_loss(self):
        fig = plt.figure()
        fig.suptitle('Loss vs Number of Epochs')
        ax = fig.add_subplot(111)
        x1 = np.arange(len(self.train_loss))
        x2 = np.arange(len(self.val_loss))
        ax.plot(x1, self.train_loss, label='Train Data')
        ax.plot(x2, self.val_loss, label='Validation Data')
        ax.set_ylabel('MSE Loss')
        ax.set_xlabel('Epoch Number')
        plt.legend()
        plt.show()

    def compute_scores(self,query_embedding,item_embedding,similarity_mesure='dot'):
        """
            Similarity Mesure: Dot Product or Cosine
        """    
        if similarity_mesure == 'dot':
            score = np.dot(query_embedding,item_embedding.T)

        elif similarity_mesure == 'cos':
            query_norm = np.linalg.norm(query_embedding)
            item_norm = np.linalg.norm(item_embedding)
            score = np.dot(query_embedding,item_embedding.T) / (query_norm * item_norm)

        return score
    
    def display_rank(self,movie_database,k_top_movies_id, k_top_scores,measure):
        movies_rank = pd.DataFrame()
        measure_key = measure + ' score'
        for i in range(len(k_top_movies_id)):
            movie_id = k_top_movies_id[i]
            movie = movie_database[movie_database['movieId']==movie_id].copy()
            movie = movie.drop('genres',axis=1)
            movie[measure_key] = k_top_scores[i]        
            movies_rank = movies_rank.append(movie)
        
        movies_rank = movies_rank.sort_values([measure_key],ascending=False)
        display(movies_rank)
    
    def user_base_rank(self,movie_database,user_id,num_recomendations=5,measure='dot'):
        user_index = -1 
        user_index = np.where(self.list_userId==user_id)
        if user_index ==  -1:
            print("Error: User Id not valid, Not Found")
            return
        
        users_embeddings = self.embedding_vars['P']
        movies_embeddings = self.embedding_vars['V']

        user_embedding = users_embeddings[user_index]
        scores = self.compute_scores(user_embedding,movies_embeddings,similarity_mesure=measure)[-1]
    
        k_top_scores_index = np.argsort(scores)[-1*num_recomendations:]
        k_top_scores = scores[k_top_scores_index]

        k_top_moviesId = self.list_movieId[k_top_scores_index]
        self.display_rank(movie_database,k_top_moviesId,np.around(k_top_scores,2),measure)
    
    def movie_base_rank(self,movie_database,movie_title,num_recomendations=5,measure='dot'):
        movies_index = movie_database[movie_database['title'].str.contains(movie_title)].index.values
        if len(movies_index) == 0:
            print("Error: " + movie_title)
            print("Movie Title invalid, Not Found")
            return
    
        movie_index = movies_index[0]
        movies_embeddings = self.embedding_vars['V']
        movie_embedding = movies_embeddings[movie_index]

        scores = self.compute_scores(movie_embedding,movies_embeddings,similarity_mesure=measure)

        k_nearest_scores_index = np.argsort(scores)[-1*num_recomendations:]
        k_nearest_scores = scores[k_nearest_scores_index]

        k_nearest_moviesId = self.list_movieId[k_nearest_scores_index]

        self.display_rank(movie_database,k_nearest_moviesId,np.around(k_nearest_scores,2),measure)