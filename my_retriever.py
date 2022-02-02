import math

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        
        # If tfidf term weighing is chosen, precompute inverse doc frequency 
        # and document vector sizes 
        if(term_weighting == "tfidf"):
            self.inverse_doc_frequency = self.inverse_doc_frequency()
            self.vector_space = self.get_vector_space()
        # If tf weighting is chosen, compute the vector sizes using term frequency
        elif(term_weighting == "tf"):
            self.vector_space = self.get_vector_space_tf()


    # Method to compute the number of documents 
    # Returns an int value of the document count
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        #Convert the query into a dictionary of term frequencies
        self.query_dic = self.convert_query_to_dic(query)
        
        #Check for term weighint selection, and run chosen method to find similar documents
        if(self.term_weighting == 'binary'):
            self.similarity_dic = self.binary_compare()
        elif (self.term_weighting == 'tf'):
            self.similarity_dic = self.tf_compare()
        else: 
            self.similarity_dic = self.tf_idf_compare()
            
            #There is another version where i include queries tfidf value in calculation
            #Chose to use the version without, as its a little quicker, and its unnecessary
            #This is in reference to note 5 of the assignment pdf
            
            #self.comparison_dic = self.tf_idf_compare_alt()

        #Order the similarity dictionary, from larger to small 
        ordered = sorted(self.similarity_dic, key=lambda x:self.similarity_dic[x], reverse=True)
        return ordered[:10]


    # Method to compute inverse document frequence for each term in the index
    # Returns a dictionary which maps terms to their inverse document frequency
    def inverse_doc_frequency(self):
        inverse_df_dic = {}
        for word in self.index:
            # Get the document frequency
            doc_f = len(self.index[word])
            inverse_df_dic[word] = math.log10(self.num_docs / doc_f)
        return inverse_df_dic
    
    # Method to compute the document vector size for each document in the index
    # Using term frequencies
    # Returns a dictionary which maps doc_ids to its squared vector size
    # Still needs to be square rooted when being used
    def get_vector_space_tf(self):
        vector_space = {}
        for word in self.index:
            for doc in self.index[word]:
                if doc in vector_space:
                    vector_space[doc] += self.index[word][doc] ** 2
                else:
                    vector_space[doc] = self.index[word][doc] ** 2
        return vector_space
    
    
    # Method to compute the document vector size for each document in the index
    # Using tf.idf
    # Returns a dictionary which maps doc_ids to its squared vector size
    # Still needs to be square rooted when being used
    def get_vector_space(self):
        vector_space = {}
        for word in self.index:
            for doc in self.index[word]:
                if doc in vector_space:
                    vector_space[doc] += (self.index[word][doc] * self.inverse_doc_frequency[word]) ** 2
                else:
                    vector_space[doc] = (self.index[word][doc] * self.inverse_doc_frequency[word]) ** 2
        return vector_space
    
    # Method to convert query from list form to dictionary of term frequencies
    # Returns a dictionary mapping term to its term frequency
    def convert_query_to_dic(self, query):
        query_dic = {}
        for word in query:
            if word not in query_dic:
                query_dic[word] = 1
            else:
                query_dic[word] += 1
        return query_dic
    
    
    # Method to compare the query to all the documents using binary weightings
    # Returns a dictionary mapping doc_id to its binary similarity measure
    def binary_compare(self):
        similarity_dic = {}
        for doc_id in self.doc_ids:
            total = 0
            for word in self.query_dic:
                if word in self.index:
                    #Adds +1 for every term which document and query both have
                    if doc_id in self.index[word]:
                        total += 1
            similarity_dic[doc_id] = total
        return similarity_dic
    
    # Method to compare the query to all the documents using tf weightings
    # Returns a dictionary mapping doc_id to its tf cosine similarity measure
    def tf_compare(self):
        similarity_dic = {}
        for doc_id in self.doc_ids:
            total = 0
            for word in self.query_dic:
                if word in self.index:
                    #If the term is in document and query, sum up the multiple of queries tf and documents tf
                    if doc_id in self.index[word]:
                        total += self.query_dic[word] * self.index[word][doc_id]
            #Get the documents tf vector size, and square root it, as the vector space dic stores the squared value
            doc_tf = math.sqrt(self.vector_space[doc_id])
            similarity_dic[doc_id] = (total/doc_tf)
        return similarity_dic
    
    # Method to compare the query to all the documents using tf_idf weightings
    # Returns a dictionary mapping doc_id to its tf_idf cosine similarity measure
    # This version doesn't bother including the queries tf_idf value, as it is the same for all
    def tf_idf_compare(self):
        similarity_dic = {}
        for doc_id in self.doc_ids:
            total = 0
            for word in self.query_dic:
                if word in self.index:
                    if doc_id in self.index[word]:
                        #Calculate the queries tf_idf value for that term
                        q_tf_idf = self.query_dic[word] * self.inverse_doc_frequency[word]
                        #Then calculate the documents tf_idf value for that term
                        word_doc_tf_idf = self.index[word][doc_id] * self.inverse_doc_frequency[word]
                        total += q_tf_idf * word_doc_tf_idf        
            #Get the documents tf_idf vector size, and square root it, as the vector space dic stores the squared value
            document_tf_idf = math.sqrt(self.vector_space[doc_id])
            similarity_dic[doc_id] = (total)/(document_tf_idf)
        return similarity_dic
        
    # This tf_idf compare method makes use of calculating the queries tf_idf value
    # This method isn't used though because it isn't needed to order the returned values
    def tf_idf_compare_alt(self):
        similarity_dic = {}
        for doc_id in self.doc_ids:
            total = 0
            for word in self.query_dic:
                if word in self.inverse_doc_frequency:
                    q_tf_idf = self.query_dic[word] * self.inverse_doc_frequency[word]
    
                    if doc_id in self.index[word]:
                        word_doc_tf_idf = self.index[word][doc_id] * self.inverse_doc_frequency[word]
                        total += q_tf_idf * word_doc_tf_idf
                        
            #Need to square root because I haven't got it square rooted in vector space
            document_tf_idf = math.sqrt(self.vector_space[doc_id])
            query_tf_idf = self.calculate_query_tf_idf(self.query_dic)
            similarity_dic[doc_id] = (total)/(query_tf_idf * document_tf_idf)
        return similarity_dic
    
    # Method which calculates a queries tf_idf value
    # Returns the tf_idf value    
    # This method is also not used, will only be needed if the alternate tf_idf compare is used
    def calculate_query_tf_idf(self, query_dic):
        tf_idf = 0
        for word in query_dic:
            if word in self.inverse_doc_frequency:
                tf_idf += (query_dic[word] * self.inverse_doc_frequency[word]) ** 2
        tf_idf = math.sqrt(tf_idf)
        return tf_idf

            
            

            
        
