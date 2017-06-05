import tensorflow as tf
import re
import codecs
import numpy as np
import math

QAFile = codecs.open("SegmentedTestQA.txt", "r", "utf-8")


Words = codecs.open("words.txt",'r', "utf-8").readlines()
WordCount = len(Words)
WordsDict = {Words[i].split()[0] : i for i in range(WordCount)}
del Words

problem_size_of_batch = 100

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
CurrentQCount = 0
DistinctQCount = 0
RelativeQCount = 0
RelativeACount = 0
LineCount = 0
BatchCount = 0
QVector.append(list())
AVector.append(list())
for word in Q:
    try:
        QVector[BatchCount].append([RelativeQCount, WordsDict[word]])
    except:
        continue
ACount.append(0)
while QALine != "":
    Temp = QALine.split("QATab")
    if (LineCount % 10000 == 0):
        print(LineCount)
    if CurrentQ != Temp[0]:
        CurrentQ = Temp[0]
        Q = Temp[0].split()
        QCount.append(CurrentQCount)
        ACount.append(LineCount)
        CurrentQCount = 1
        DistinctQCount = DistinctQCount + 1
        if DistinctQCount % problem_size_of_batch == 0:
            BatchCount += 1
            QVector.append(list())
            AVector.append(list())
            RelativeQCount = 0
            RelativeACount = 0
        for word in Q:
            try:
                QVector[BatchCount].append([RelativeQCount, WordsDict[word]])
            except:
                continue
    else:
        CurrentQCount = CurrentQCount + 1
    A = Temp[1].split()
    for word in A:
        try:
            AVector[BatchCount].append([RelativeACount, WordsDict[word]])
        except:
            continue
    LineCount = LineCount + 1
    RelativeACount += 1
    QALine = QAFile.readline()
QCount.append(CurrentQCount)
ACount.append(LineCount)
DistinctQCount = DistinctQCount + 1
print("Loading finished")
'''
FlagIndex = 0
Temp = Flags
Flags = list()
for i in range(DistinctQCount):
    AnswerCount = 0
    Flags.append(list())
    while not FlagIndex == ACount[i + 1]:
        if Temp[FlagIndex] == "1\r\n":
            Flags[i].append(AnswerCount)
        FlagIndex += 1
        AnswerCount += 1
del Temp, FlagIndex
'''

QHolder = tf.sparse_placeholder(tf.float32, [None, WordCount])
AHolder = tf.sparse_placeholder(tf.float32, [None, WordCount])
QCountHolder = tf.placeholder(tf.int32, [None])
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

cond = lambda k, logs, CulAIndex, LoopTimesHolder, Q, A : tf.less(k, LoopTimesHolder[0])
def body(k, logs, CulAIndex, LoopTimesHolder, Q, A):
    product = tf.reshape(tf.matmul([Q[k]], A[CulAIndex:CulAIndex + QCountHolder[k], :], transpose_b = True), [-1, 1])
    norm_product = norm_Q[k] * norm_A[CulAIndex:CulAIndex + QCountHolder[k], :]
    #logs = tf.concat([logs, tf.reshape(tf.truediv(product, norm_product), [-1])], 0)
    logs = tf.concat([logs, tf.reshape(product, [-1])], 0)
    CulAIndex += QCountHolder[k]
    k += 1
    return [k, logs, CulAIndex, LoopTimesHolder, Q, A]

CulAIndex = tf.constant(0, tf.int32)
logs = tf.convert_to_tensor([], tf.float32)
i = tf.constant(0, tf.int32)
f = tf.while_loop(cond, body, [i, logs, CulAIndex, LoopTimesHolder, Q, A], shape_invariants=[i.get_shape(), tf.TensorShape([None,]), CulAIndex.get_shape(), LoopTimesHolder.get_shape(), Q.get_shape(), A.get_shape()])
'''
for k in range(problem_size_of_batch):
    product = tf.reshape(tf.matmul([Q[k]], A[CulAIndex:CulAIndex + QCountHolder[k], :], transpose_b = True), [-1, 1])
    norm_product = norm_Q[k] * norm_A[CulAIndex:CulAIndex + QCountHolder[k], :]
    temp = -tf.log(tf.reshape(tf.nn.softmax(tf.reshape(tf.truediv(product, norm_product), [-1])), [1, -1]))
    logs += tf.matmul(temp, FlagsHolder[CulAIndex:CulAIndex + QCountHolder[k], :])
    CulAIndex += QCountHolder[k]
loss = logs
train = tf.train.GradientDescentOptimizer(1).minimize(loss)
if (not DistinctQCount % problem_size_of_batch == 0):
    logs2 = tf.constant(0, tf.float32)
    for k in range(DistinctQCount % problem_size_of_batch):
        product = tf.reshape(tf.matmul([Q[k]], A[CulAIndex:CulAIndex + QCountHolder[k], :], transpose_b = True), [-1, 1])
        norm_product = norm_Q[k] * norm_A[CulAIndex:CulAIndex + QCountHolder[k], :]
        temp = -tf.log(tf.reshape(tf.nn.softmax(tf.reshape(tf.truediv(product, norm_product), [-1])), [1, -1]))
        logs2 += tf.matmul(temp, FlagsHolder[CulAIndex:CulAIndex + QCountHolder[k], :])
        CulAIndex += QCountHolder[k]
    loss2 = logs2
    train2 = tf.train.GradientDescentOptimizer(1).minimize(loss2)
    '''
print("Graph initialize finish")
# get Q and tile it, then matmul to get cos_sim '''
#assume always have 1
'''
Q = [Q[0]]
A = [A[0], A[1]]
norm_Q = norm_Q[0]
norm_A = [norm_A[0], norm_A[1]]
product = tf.matmul(Q, tf.transpose(A))
product = tf.reshape(product, [-1, 1])
norm_product = norm_Q * norm_A
cos_sim = tf.nn.softmax(tf.reshape(tf.truediv(product, norm_product), [-1]))
loss = -tf.reduce_sum(tf.log(cos_sim[0]))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
'''
resFile = open("testResult.txt", "w")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "model")
    for i in range(1):
        for j in range(len(QVector)): #len of QVector = count of batches
            if j == len(QVector) - 1 and not DistinctQCount % problem_size_of_batch == 0:
                CurrentSize = DistinctQCount % problem_size_of_batch
            else:
                CurrentSize = problem_size_of_batch
            CurrentBatch = problem_size_of_batch * j
            NextBatch = CurrentBatch + CurrentSize
            QTrain = tf.SparseTensorValue(QVector[j], [1 for _ in range(len(QVector[j]))], [CurrentSize, WordCount])
            ATrain = tf.SparseTensorValue(AVector[j], [1 for _ in range(len(AVector[j]))], [ACount[NextBatch] - ACount[CurrentBatch], WordCount]) 
            feed_dict = {QHolder: QTrain, AHolder: ATrain, QCountHolder: QCount[CurrentBatch:NextBatch], LoopTimesHolder: [CurrentSize]}
            cos_sim = np.array(sess.run([f[1]], feed_dict))
            for item in cos_sim[0]:
                resFile.write(str(item) + "\r\n")
resFile.close()
