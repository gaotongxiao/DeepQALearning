import tensorflow as tf
import re
import codecs
import numpy as np
import math

QAFile = codecs.open("SegmentedQA.txt", "r", "utf-8")
FlagFile = codecs.open("Flag.txt", "r", "utf-8")

All_Flags = list()

Words = codecs.open("words.txt",'r', "utf-8").readlines()
WordCount = len(Words)
WordsDict = {Words[i].split()[0] : i for i in range(WordCount)}
del Words

Flags = (codecs.open("Flag.txt", "rU", "utf-8").readlines())

problem_size_of_batch = 200

#assume that there is only 1 problem
QVector = list()
AVector = list()
QCount = list() #store the number of answers to specific problem
ACount = list() #store the lines of answers to specific problem (start: ACount[n] End: Acount[n+1])

WordsIndex = 0
QALine = QAFile.readline()
Temp = QALine.split("QATab")
CurrentQ = Temp[0]
Q = Temp[0].split()
CurrentAnswerCount = 0 #to count the number of answers of current question
DistinctQCount = 0#to count the number of distinct questions
LineCount = 0 #to count the line # of current line
RelativeQCount = 0 #represent the current Q index relative to the batch size
RelatvieACount = 0 # similiarly
BatchCount = 0
QVector.append(list())
AVector.append(list())
for word in Q:
    QVector[BatchCount].append([RelativeQCount, WordsDict[word]])
ACount.append(0)
while QALine != "":
    Temp = QALine.split("QATab")
    if (LineCount % 10000 == 0):
        print(LineCount)
    if CurrentQ != Temp[0]:
        CurrentQ = Temp[0]
        Q = Temp[0].split()
        QCount.append(CurrentAnswerCount)
        ACount.append(LineCount)
        CurrentAnswerCount = 1
        DistinctQCount = DistinctQCount + 1
        if DistinctQCount % problem_size_of_batch == 0:
            BatchCount += 1
            QVector.append(list())
            AVector.append(list())
            RelativeQCount = 0
            RelatvieACount = 0
        else:
            RelativeQCount += 1
        for word in Q:
            QVector[BatchCount].append([RelativeQCount, WordsDict[word]])
    else:
        CurrentAnswerCount = CurrentAnswerCount + 1
    A = Temp[1].split()
    for word in A:
        AVector[BatchCount].append([RelatvieACount, WordsDict[word]])
    LineCount = LineCount + 1
    RelatvieACount += 1
    QALine = QAFile.readline()
QCount.append(CurrentAnswerCount)
ACount.append(LineCount)
DistinctQCount = DistinctQCount + 1
print(QCount[len(QVector) -1:len(QVector)])
print("Loading finished")

Flags = [[int(f)] for f in Flags]

QHolder = tf.sparse_placeholder(tf.float32, [None, WordCount])
AHolder = tf.sparse_placeholder(tf.float32, [None, WordCount])
QCountHolder = tf.placeholder(tf.int32, [None])
FlagsHolder = tf.placeholder(tf.float32, [None, 1])
LoopTimesHolder = tf.placeholder(tf.int32, [1])

print("Training Dataset initialized")

L1_N = 400
L2_N = 400
L3_N = 128


L1_range = np.sqrt(6.0 / (WordCount + L1_N))
W1 = tf.Variable(tf.random_uniform([WordCount, L1_N], -L1_range, L1_range), name = 'W1')
b1 = tf.Variable(tf.random_uniform([L1_N], -L1_range, L1_range), name = 'b1')
Q = tf.sparse_tensor_dense_matmul(QHolder, W1) + b1
A = tf.sparse_tensor_dense_matmul(AHolder, W1) + b1
Q = tf.nn.relu(Q)
A = tf.nn.relu(A)

L2_range = np.sqrt(6.0 / (L1_N + L2_N))
W2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -L2_range, L2_range), name = 'W2')
b2 = tf.Variable(tf.random_uniform([L2_N], -L2_range, L2_range), name = 'b2')
Q = tf.matmul(Q, W2) + b2
A = tf.matmul(A, W2) + b2
Q = tf.nn.relu(Q)
A = tf.nn.relu(A)

L3_range = np.sqrt(6.0 / (L2_N + L3_N))
W3 = tf.Variable(tf.random_uniform([L2_N, L3_N], -L3_range, L3_range), name = 'W3')
b3 = tf.Variable(tf.random_uniform([L3_N], -L3_range, L3_range), name = 'b3')
Q = tf.matmul(Q, W3) + b3
A = tf.matmul(A, W3) + b3
Q = tf.nn.relu(Q)
A = tf.nn.relu(A)


saver = tf.train.Saver({"W1":W1, "b1":b1, "W2":W2, "b2":b2, "W3":W3, "b3":b3})
'''
Q [ALL_Size, L3_N]
A [ALL_Size, L3_N]
norm_Q: [ALL_Size, 1]
norm_A: [ALL_Size, 1]
'''

norm_Q = tf.reduce_sum(tf.sqrt(Q), 1, keep_dims = True)
norm_A = tf.reduce_sum(tf.sqrt(A), 1, keep_dims = True)

cond = lambda k, logs, CulAIndex, LoopTimesHolder : tf.less(k, LoopTimesHolder[0])
def body(k, logs, CulAIndex, LoopTimesHolder):
    product = tf.reshape(tf.matmul([Q[k]], A[CulAIndex:CulAIndex + QCountHolder[k], :], transpose_b = True), [-1, 1])
    norm_product = norm_Q[k] * norm_A[CulAIndex:CulAIndex + QCountHolder[k], :]
    temp = -tf.log(tf.reshape(tf.nn.softmax(tf.reshape(tf.truediv(product, norm_product), [-1])), [1, -1]))
    logs += tf.matmul(temp, FlagsHolder[CulAIndex:CulAIndex + QCountHolder[k], :])[0][0]
    CulAIndex += QCountHolder[k]
    k += 1
    return [k, logs, CulAIndex, LoopTimesHolder]

CulAIndex = 0
logs = tf.constant(0, tf.float32)
f = tf.while_loop(cond, body, [tf.constant(0, tf.int32), logs, CulAIndex, LoopTimesHolder])
loss = f[1]
globalstep = tf.Variable(0, trainable = False)
learning_rate = tf.train.exponential_decay(0.03, globalstep, 1000, decay_rate = 0.8)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = globalstep)
print("Graph initialize finish")
# get Q and tile it, then matmul to get cos_sim '''
#assume always have 1
try:
    bestovl = float(re.sub("\r\n", "", open("CostRecord.txt", "r").readlines()[-1]))
except:
    bestovl = 10000000
print(bestovl)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if raw_input("1. model 2. modelstep") == "1":
        print("restoring model")
        saver.restore(sess, "model")
    else:
        print("resotring modelstep")
        saver.restore(sess, "modelstep")
    for i in range(1000):
        ovl = 0
        for j in range(len(QVector)): #len of QVector = count of batches
            if j == len(QVector) - 1 and not DistinctQCount % problem_size_of_batch == 0:
                CurrentSize = DistinctQCount % problem_size_of_batch
            else:
                CurrentSize = problem_size_of_batch
            CurrentBatch = problem_size_of_batch * j
            NextBatch = CurrentBatch + CurrentSize
            QTrain = tf.SparseTensorValue(QVector[j], [1 for _ in range(len(QVector[j]))], [CurrentSize, WordCount])
            ATrain = tf.SparseTensorValue(AVector[j], [1 for _ in range(len(AVector[j]))], [ACount[NextBatch] - ACount[CurrentBatch], WordCount]) 
            feed_dict = {QHolder: QTrain, AHolder: ATrain, QCountHolder: QCount[CurrentBatch:NextBatch], FlagsHolder: Flags[ACount[CurrentBatch]:ACount[NextBatch]], LoopTimesHolder: [CurrentSize]}
            l, _ = sess.run([loss, train], feed_dict)
            ovl += l
            if (j==0):
                print(l)
        print(ovl)
        if ovl < bestovl:
            saver.save(sess, "model")
            try:
                FileRecord.write(str(ovl) + "\r\n")
            except:
                FileRecord = open("CostRecord.txt", "w")
                FileRecord.write(str(ovl) + "\r\n")
            bestovl = ovl
            print("good")
        else:
            saver.save(sess, "modelstep")
