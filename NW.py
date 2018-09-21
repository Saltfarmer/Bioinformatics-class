import numpy as np
import random

MISMATCH = -6;
GAP      = -6;
BLOSUM62={'C':{'C':9,'S':-1,'T':-1,'P':-3,'A':0,'G':-3,'N':-3,'D':-3,'E':-4,'Q':-3,'H':-3,'R':-3,'K':-3,'M':-1,'I':-1,'L':-1,'V':-1,'F':-2,'Y':-2,'W':-2},
					'S':{'C':-1,'S':4,'T':1,'P':-1,'A':1,'G':0,'N':1,'D':0,'E':0,'Q':0,'H':-1,'R':-1,'K':0,'M':-1,'I':-2,'L':-2,'V':-2,'F':-2,'Y':-2,'W':-3},
					'T':{'C':-1,'S':1,'T':4,'P':1,'A':-1,'G':1,'N':0,'D':1,'E':0,'Q':0,'H':0,'R':-1,'K':0,'M':-1,'I':-2,'L':-2,'V':-2,'F':-2,'Y':-2,'W':-3},
					'P':{'C':-3,'S':-1,'T':1,'P':7,'A':-1,'G':-2,'N':-1,'D':-1,'E':-1,'Q':-1,'H':-2,'R':-2,'K':-1,'M':-2,'I':-3,'L':-3,'V':-2,'F':-4,'Y':-3,'W':-4},
					'A':{'C':0,'S':1,'T':-1,'P':-1,'A':4,'G':0,'N':-1,'D':-2,'E':-1,'Q':-1,'H':-2,'R':-1,'K':-1,'M':-1,'I':-1,'L':-1,'V':-2,'F':-2,'Y':-2,'W':-3},
					'G':{'C':-3,'S':0,'T':1,'P':-2,'A':0,'G':6,'N':-2,'D':-1,'E':-2,'Q':-2,'H':-2,'R':-2,'K':-2,'M':-3,'I':-4,'L':-4,'V':0,'F':-3,'Y':-3,'W':-2},
					'N':{'C':-3,'S':1,'T':0,'P':-2,'A':-2,'G':0,'N':6,'D':1,'E':0,'Q':0,'H':-1,'R':0,'K':0,'M':-2,'I':-3,'L':-3,'V':-3,'F':-3,'Y':-2,'W':-4},
					'D':{'C':-3,'S':0,'T':1,'P':-1,'A':-2,'G':-1,'N':1,'D':6,'E':2,'Q':0,'H':-1,'R':-2,'K':-1,'M':-3,'I':-3,'L':-4,'V':-3,'F':-3,'Y':-3,'W':-4},
					'E':{'C':-4,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':2,'E':5,'Q':2,'H':0,'R':0,'K':1,'M':-2,'I':-3,'L':-3,'V':-3,'F':-3,'Y':-2,'W':-3},
					'Q':{'C':-3,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':0,'E':2,'Q':5,'H':0,'R':1,'K':1,'M':0,'I':-3,'L':-2,'V':-2,'F':-3,'Y':-1,'W':-2},
					'H':{'C':-3,'S':-1,'T':0,'P':-2,'A':-2,'G':-2,'N':1,'D':1,'E':0,'Q':0,'H':8,'R':0,'K':-1,'M':-2,'I':-3,'L':-3,'V':-2,'F':-1,'Y':2,'W':-2},
					'R':{'C':-3,'S':-1,'T':-1,'P':-2,'A':-1,'G':-2,'N':0,'D':-2,'E':0,'Q':1,'H':0,'R':5,'K':2,'M':-1,'I':-3,'L':-2,'V':-3,'F':-3,'Y':-2,'W':-3},
					'K':{'C':-3,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':-1,'E':1,'Q':1,'H':-1,'R':2,'K':5,'M':-1,'I':-3,'L':-2,'V':-3,'F':-3,'Y':-2,'W':-3},
					'M':{'C':-1,'S':-1,'T':-1,'P':-2,'A':-1,'G':-3,'N':-2,'D':-3,'E':-2,'Q':0,'H':-2,'R':-1,'K':-1,'M':5,'I':1,'L':2,'V':-2,'F':0,'Y':-1,'W':-1},
					'I':{'C':-1,'S':-2,'T':-2,'P':-3,'A':-1,'G':-4,'N':-3,'D':-3,'E':-3,'Q':-3,'H':-3,'R':-3,'K':-3,'M':1,'I':4,'L':2,'V':1,'F':0,'Y':-1,'W':-3},
					'L':{'C':-1,'S':-2,'T':-2,'P':-3,'A':-1,'G':-4,'N':-3,'D':-4,'E':-3,'Q':-2,'H':-3,'R':-2,'K':-2,'M':2,'I':2,'L':4,'V':3,'F':0,'Y':-1,'W':-2},
					'V':{'C':-1,'S':-2,'T':-2,'P':-2,'A':0,'G':-3,'N':-3,'D':-3,'E':-2,'Q':-2,'H':-3,'R':-3,'K':-2,'M':1,'I':3,'L':1,'V':4,'F':-1,'Y':-1,'W':-3},
					'F':{'C':-2,'S':-2,'T':-2,'P':-4,'A':-2,'G':-3,'N':-3,'D':-3,'E':-3,'Q':-3,'H':-1,'R':-3,'K':-3,'M':0,'I':0,'L':0,'V':-1,'F':6,'Y':3,'W':1},
					'Y':{'C':-2,'S':-2,'T':-2,'P':-3,'A':-2,'G':-3,'N':-2,'D':-3,'E':-2,'Q':-1,'H':2,'R':-2,'K':-2,'M':-1,'I':-1,'L':-1,'V':-1,'F':3,'Y':7,'W':2},
					'W':{'C':-2,'S':-3,'T':-3,'P':-4,'A':-3,'G':-2,'N':-4,'D':-4,'E':-3,'Q':-2,'H':-2,'R':-3,'K':-3,'M':-1,'I':-3,'L':-2,'V':-3,'F':1,'Y':2,'W':11}
					}

def random_AA_seq(length):
		result='M'
		for i in range(length-1):
				result = result+str(random.choice('ACDEFGHIKLMNPQRSTVWY'))
		return result

x_ax=[]
number_of_aligments = 1
length = 10

for i in range(0, number_of_aligments): 
		sequence1 = random_AA_seq(length)
		sequence2 = random_AA_seq(length)

		print ("Alignment:",i+1)
		print (sequence1)
		print (sequence2 + '\n')
		#initialisation
		score_matrix = np.zeros([len(sequence2)+1,len(sequence1)+1])
		trace_matrix = np.zeros([len(sequence2)+1,len(sequence1)+1],dtype=str)
		#extension penalty
		for j in range(0,len(sequence1)+1):
				score_matrix[0][j] = GAP*j
				trace_matrix[0][j] = "L"
		for i in range(0,len(sequence2)+1):
				score_matrix[i][0] = GAP*i
				trace_matrix[i][0] = "U"
		score_matrix[0][0] = 0
		trace_matrix[0][0] = "N"
		print ("Alignment Needleman-Wunsch...",end=" ")
		for i in range(1,len(sequence2)+1):
				for j in range (1,len(sequence1)+1):
						diagonal_score=0
						left_score=0
						up_score=0
						# calculate match/mismatch score
						letter1 = sequence1[j-1:j]
						letter2 = sequence2[i-1:i]        
						if (letter1 == letter2):
								diagonal_score = score_matrix[i-1][j-1] + BLOSUM62[letter1][letter2]
						else:
								diagonal_score = score_matrix[i-1][j-1] + BLOSUM62[letter1][letter2]
						# calculate gap scores
						if (trace_matrix[i-1][j] == "D"):
								up_score   = score_matrix[i-1][j] + GAP
						if (trace_matrix[i][j-1] == "D"):
								left_score = score_matrix[i][j-1] + GAP
						# choose best score
						if (diagonal_score >= up_score):
								if (diagonal_score >= left_score):
										score_matrix[i][j] = diagonal_score
										trace_matrix[i][j] = "D"
								else:
										score_matrix[i][j] = left_score
										trace_matrix[i][j] = "L"
						else:
								if (up_score >= left_score):
										score_matrix[i][j] = up_score
										trace_matrix[i][j] = "U"
								else:
										score_matrix[i][j] = left_score
										trace_matrix[i][j] = "L"
										 
		print ("\n Score Matrix:")
		print (score_matrix)
		print ("\n Trace Matrix:")
		print (trace_matrix)
		align1 = ""
		align2 = ""
		tracking = ""
		j = len(sequence1)
		i = len(sequence2)
		#print ("Backtracking:",end="\n")
		tracking_score = []
		while trace_matrix[i][j] != "N":
				tracking = tracking + trace_matrix[i][j]
				tracking_score.append(score_matrix[i][j])
				if (trace_matrix[i][j] == "D"):
						align1 = align1 + sequence1[j-1:j]
						align2 = align2 + sequence2[i-1:i]
						i=i-1
						j=j-1
				elif (trace_matrix[i][j] == "L"):
						align1 = align1 + sequence1[j-1:j]
						align2 = align2 + "-"
						j=j-1
				elif (trace_matrix[i][j] == "U"):
						align1 = align1 + "-"
						align2 = align2 + sequence2[i-1:i]
						i=i-1
		print ("\n\n\n\n")
		print ("Scoring - Backtrace - Alignment:",end="\n")
		align1 = align1[::-1]
		align2 = align2[::-1]
		tracking = tracking[::-1]
		print (list(reversed(tracking_score)))
		print (tracking)
		print (align1)
		print (align2)                
		print (" score:",score_matrix[len(sequence2)][len(sequence1)])
		x_ax.append(score_matrix[len(sequence2)][len(sequence1)])