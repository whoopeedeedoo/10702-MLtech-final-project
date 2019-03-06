import time, pandas as pd, numpy as np

def main():
	time_start = time.time()
	csvs={'books': 'data/books.csv'}
	books = pd.read_csv(csvs['books'], ',').values
	# example: 
	# ISBN,Book-Title,Book-Author,Year-Of-Publication,Publisher,Image-URL-S,Image-URL-M,Image-URL-L,Book-Description
	# shape = (271379, 9)
	print(books[0, 8])
	time_end = time.time()
	print('execution time:', time_end - time_start, 's')
	print('program ends')	
main()