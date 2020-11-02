# Font KNN

################################ Data Cleaning #################################

attach(BELL)
library(dplyr) # Package for subseting data
bell_clean <- select(BELL,-c(2,3,6,7,8,9,10,11,12)) # Discard the 9 columns
# 956x403

attach(CHILLER)
chiller_clean <- select(CHILLER,-c(2,3,6,7,8,9,10,11,12))
# 952x403

attach(FREESTYLE)
freestyle_clean <- select(FREESTYLE,-c(2,3,6,7,8,9,10,11,12))
# 956x403

# Discarding row containing missing numerical data
BELL_clean <- na.omit(bell_clean)
CHILLER_clean <- na.omit(chiller_clean)
FREESTYLE_clean <- na.omit(freestyle_clean)

# Defining three classes of images of normal characters
# cl1 = all rows of BELL_clean.csv file for which (strength = 0.4 and italic =0)
# cl2 = all rows of CHILLER_clean.csv file for which (strength = 0.4 and italic =0)
# cl3 = all rows of FREESTYLE_clean.csv file for which (strength = 0.4 and italic =0)

BELL_clean <- data.frame(BELL_clean)#creating a data frame to add conditional statements to filter out non needed i features
BELL_clean$CL = ifelse((BELL_clean$strength == 0.4 & BELL_clean$italic == 0),"CL1","NA")
BELL_CLEAN = BELL_clean[which(BELL_clean$CL =="CL1"),] #labeling the new filter data as cl1
# 404x239

CHILLER_clean <- data.frame(CHILLER_clean)
CHILLER_clean$CL = ifelse((CHILLER_clean$strength == 0.4 & CHILLER_clean$italic == 0),"CL2","NA")
CHILLER_CLEAN = CHILLER_clean[which(CHILLER_clean$CL =="CL2"),]
# 404x238

FREESTYLE_clean <- data.frame(FREESTYLE_clean)
FREESTYLE_clean$CL = ifelse((FREESTYLE_clean$strength == 0.4 & FREESTYLE_clean$italic == 0),"CL3","NA")
FREESTYLE_CLEAN = FREESTYLE_clean[which(FREESTYLE_clean$CL =="CL3"),]
# 404x239

# Combine CL1, CL2, CL3 into DATA
DATA <- rbind(BELL_CLEAN,CHILLER_CLEAN,FREESTYLE_CLEAN)

# Binded all 3 data sets to a full data set (DATA) which is the union of 3 classes (CL1, CL2, CL3)
# where N = 716

#################################### Standardize Data ####################################

# Standardize must have mean = 1 and sd = 1

# Compute mean
DATAMEAN <- DATA %>% summarize_if(is.numeric,mean)
mean(DATA[,3]) # Mean of this column is 0 which is ok

# Compute standard deviation
DATASD <- DATA %>% summarize_if(is.numeric, sd)
var(DATA[,3]) # sd is 0 which is not okay, which we need to standardize to have a comparable scale

# Standardizing to make a comparable scale
library(standardize)
SDATA <- DATA %>% mutate_if(is.numeric, function (x) as.vector(scale(x))) # scaling by (xj-mj)/sd
SDATA = SDATA[,-c(2,3)] # Taking out numerical functions of strength and italics
sDATA <- data.matrix(SDATA) # Creating it into a data matrix for correlation matrix beforehand
## SDATA contains CL classes and font name, but not strength and italics.
## confirming the standardization has properly worked by looking at mean and sd again of the SDATA
var(SDATA[,3]) # sd = 1 which is good

# Scale the data again for the
sDATA1 <- scale(sDATA[,-c(1,402)]) # Scaling and removing non-numerical values
sDATA1
## sDATA1 is data set containing standardized features, but without non-numerical values.

################################## Correlation Matrix ##################################

cor(sDATA1)
cor.df = data.frame(cor(sDATA1)) # Renaming to view actual full matrix

# Find top 10 highest values
library(dplyr)
library(tidyr)
topvalues_sdata<-cor(sDATA1) %>%
  as.data.frame() %>%
  mutate(var1 = rownames(.)) %>%
  gather(var2, value, -var1) %>%
  arrange(desc(value)) %>%
  group_by(value) %>%
  filter(row_number()==1)

############################# Original train and test sets ############################
                            
# Loop to classify cl to SETROW columns into data set
SDATA$SETROW <- NA

for (i in 1:716){
  if(SDATA$CL[i]=="CL1"){
    SDATA$SETROW[i] = "SETROW1"
  }else if(SDATA$CL[i]=="CL2"){
    SDATA$SETROW[i] = "SETROW2"
  }else{
    SDATA$SETROW[i] = "SETROW3"
  }
}
                            
# Creating the 80% random train set interval by taking ONLY using setrow 1, we replicate this for the other setrow functions
SETROW1 = SDATA[which(SDATA$SETROW =="SETROW1"),]
n <- nrow(SETROW1[which(SETROW1$SETROW=="SETROW1"),])
trainset <- sample(1:n, 0.8*n)
trainsetcl1 <- SETROW1[trainset,]
testsetcl1 <- SETROW1[-trainset,]

SETROW2 = SDATA[which(SDATA$SETROW =="SETROW2"),]
n <- nrow(SETROW2[which(SETROW2$SETROW=="SETROW2"),])
trainset <- sample(1:n, 0.8*n)
trainsetcl2 <- SETROW2[trainset,]
testsetcl2 <- SETROW2[-trainset,]

SETROW3 = SDATA[which(SDATA$SETROW =="SETROW3"),]
n <- nrow(SETROW3[which(SETROW3$SETROW=="SETROW3"),])
trainset <- sample(1:n, 0.8*n)
trainsetcl3 <- SETROW3[trainset,]
testsetcl3 <- SETROW3[-trainset,]

# Combining the sets to full trainset and testset
TRAIN_SET <- rbind(trainsetcl1,trainsetcl2,trainsetcl3)
TEST_SET <- rbind(testsetcl1,testsetcl2,testsetcl3)

# Train and test labels
library(class)
SDATA_no <- SDATA[,-c(1,402,403)]
SDATA_label <- SDATA[,"CL"]
TRAIN_no <- TRAIN_SET[,-c(1,402,403)]
TRAIN_label <- TRAIN_SET[, "CL"]
TEST_no <- TEST_SET[,-c(1,402,403)]
TEST_label <- TEST_SET[,"CL"]

############################## Running KNN using K = 12 ##############################
                            
# Compute the percentage of correct classification
RNGkind(sample.kind = "Rounding")
set.seed(1)
knn.predtrain12 <- knn(train=TRAIN_no,
                       test=TRAIN_no,
                       cl = TRAIN_label,
                       k=12)

RNGkind(sample.kind = "Rounding")
set.seed(1)
knn.predtest12 <- knn(train=TRAIN_no,
                      test=TEST_no,
                      cl = TRAIN_label,
                      k=12)

mean(knn.predtrain12 == TRAIN_label) #0.7377622
mean(knn.predtest12 == TEST_label) #0.6805556

# Confusion matrix
table(data.frame(knn.predtrain,TRAIN_label))
table(data.frame(knn.predtest, TEST_label))

############################## Finding the Best K ##############################
                     
# Finding percent accuracy for each value of 5,10...100
set.seed(1)
i=1
k.optm=1
for (i in seq(5, 100, by = 5)){
  knn.mod<- knn(train = TRAIN_no, test = TEST_no, cl = TRAIN_label, k= i)
  k.optm[i]<- 100 * sum(TEST_label == knn.mod)/ NROW(TEST_label)
  k=i
  cat(k,"=", k.optm[i],'\n')
}

############################ Plot of Accuracy vs K ##############################                           
                            
# Fit the model on the training set finding the optimized value of k for test set
# Running a loop to find accuracy for test set
set.seed(1)
K.set = c(5,10,15,20,25,30,40,50,100)
knn.test.accuracy <- numeric(length(K.set))
for (j in 1:length(K.set)){
  knn.pred <- knn(train=TRAIN_no,
                  test=TEST_no,
                  cl=TRAIN_label,
                  k=K.set[j])
  knn.test.accuracy[j] <- mean(knn.pred == TEST_label)
}
# Finding accuracy for train set which will be higher than test set
set.seed(1)
K.set = c(5,10,15,20,25,30,40,50,100)
knn.train.accuracy <- numeric(length(K.set))

for (j in 1:length(K.set)){
  knn.pred <- knn(train=TRAIN_no,
                  test=TRAIN_no,
                  cl=TRAIN_label,
                  k=K.set[j])
  knn.train.accuracy[j] <- mean(knn.pred == TRAIN_label)
}
# Plot the accuracy vs K of train and test sets
# red = test set    blue = train set
plot(K.set, knn.train.accuracy, type="o", col="blue", pch="o", lty=1 )
points(K.set, knn.test.accuracy, col="red", pch="*")
lines(K.set, knn.test.accuracy, col="red",lty=2)


######################### KNN Before subsetting PACKS 1-4 #########################                  
                   
# Applying the "best" k value to the both train and test set.
set.seed(1)
knn.predtrainbest<- knn(train=TRAIN_no,
                        test=TRAIN_no,
                        cl = TRAIN_label,
                        k=5)
set.seed(1)
knn.predtestbest <- knn(train=TRAIN_no,
                        test=TEST_no,
                        cl = TRAIN_label,
                        k=5)
# Displaying the percent acc
mean(knn.predtrainbest == TRAIN_label)
mean(knn.predtestbest == TEST_label)

# Displaying confusion matrix of cl 
trainmt <- table(data.frame(knn.predtrainbest,TRAIN_label))
testtt <- table(data.frame(knn.predtestbest, TEST_label))

# Finding confidence intervals of confusion matrix
library(DescTools)
Conf(trainmt)
#yielding 95% CI : (0.7975, 0.8589)
Conf(testtt)
#yielding 95% CI : (0.6734, 0.8136)

############################ Create subsets for PACKS 1-4 ############################
                            
# PACK 1 L: 0-9 and M: 0-9 making a 100 attributes
PACK1 <- SDATA[,c(2:11,22:31,42:51,62:71,82:91,102:111,122:131,142:151,162:171,182:191,402,403)]
PACK2 <- SDATA[,c(12:21,32:41,52:61,72:81,92:101,112:121,132:141,152:161,172:181,192:201,402,403)]
PACK3 <- SDATA[,c(212:221,232:241,252:261,272:281,292:301,312:321,332:341,352:361,372:381,392:401,402,403)]
PACK4 <- SDATA[,c(202:211,222:231,242:251,262:271,282:291,302:311,322:331,342:351,362:371,382:391,402,403)]

# Pack 1 CL1
# Dividing set of pack 1 of .8 of train .2 test
packcl1 = PACK1[which(PACK1$SETROW=="SETROW1"),]
n <- nrow(packcl1)
PACKCL11 <- sample(1:n, 0.8*n)
PACKCL1_train1 <- packcl1[PACKCL11,]
PACKCL1_test1 <- packcl1[-PACKCL11,]

# Replicating for CL2,
packcl2 = PACK1[which(PACK1$SETROW=="SETROW2"),]
n <- nrow(packcl2)
PACKCL21 <- sample(1:n, 0.8*n)
PACKCL2_train1 <- packcl2[PACKCL21,]
PACKCL2_test1 <- packcl2[-PACKCL21,]

# Replicating for CL3
packcl3 = PACK1[which(PACK1$SETROW=="SETROW3"),]
n <- nrow(packcl3)
PACKCL31 <- sample(1:n, 0.8*n)
PACKCL3_train1 <- packcl3[PACKCL31,]
PACKCL3_test1 <- packcl3[-PACKCL31,]
                            
# UNIONIZING PACK 1 CLs
PACK1_TRAINALL <- rbind(PACKCL1_train1,PACKCL2_train1,PACKCL3_train1)
PACK1_TESTALL <- rbind(PACKCL1_test1,PACKCL2_test1,PACKCL3_test1)

# PACK 2
# Pack CL1
# Dividing set of pack 1 of .8 of train .2 test
packcl1p2 = PACK2[which(PACK2$SETROW=="SETROW1"),]
n <- nrow(packcl1p2)
PACKCL1p2 <- sample(1:n, 0.8*n)
PACKCL1_train2 <- packcl1p2[PACKCL1p2,]
PACKCL1_test2 <- packcl1p2[-PACKCL1p2,]

# Pack2 CL2
packcl2p2 = PACK2[which(PACK2$SETROW=="SETROW2"),]
n <- nrow(packcl2p2)
PACKCL2p2 <- sample(1:n, 0.8*n)
PACKCL2_train2 <- packcl2p2[PACKCL2p2,]
PACKCL2_test2 <- packcl2p2[-PACKCL2p2,]

# Pack 2 CL3
packcl3p2 = PACK2[which(PACK2$SETROW=="SETROW1"),]
n <- nrow(packcl3p2)
PACKCL3p2 <- sample(1:n, 0.8*n)
PACKCL3_train2 <- packcl3p2[PACKCL3p2,]
PACKCL3_test2 <- packcl3p2[-PACKCL3p2,]

# UNIONIZING PACK 2 CLs
PACK2_TRAINALL <- rbind(PACKCL1_train2,PACKCL2_train2,PACKCL3_train2)
PACK2_TESTALL <- rbind(PACKCL1_test2,PACKCL2_test2,PACKCL3_test2)

# PACK 3
# Pack 3 CL1
packcl1p3 = PACK3[which(PACK3$SETROW=="SETROW1"),]
n <- nrow(packcl1p3)
PACKCL1p3 <- sample(1:n, 0.8*n)
PACKCL1_train3 <- packcl1p3[PACKCL1p3,]
PACKCL1_test3 <- packcl1p3[-PACKCL1p3,]

# CL2
packcl2p3 = PACK3[which(PACK3$SETROW=="SETROW2"),]
n <- nrow(packcl2p3)
PACKCL2p3 <- sample(1:n, 0.8*n)
PACKCL2_train3 <- packcl2p3[PACKCL2p3,]
PACKCL2_test3 <- packcl2p3[-PACKCL2p3,]

# CL3
packcl3p3 = PACK3[which(PACK3$SETROW=="SETROW3"),]
n <- nrow(packcl3p3)
PACKCL3p3 <- sample(1:n, 0.8*n)
PACKCL3_train3 <- packcl3p3[PACKCL3p3,]
PACKCL3_test3 <- packcl3p3[-PACKCL3p3,]

# UNIONIZING PACK 3 CLs
PACK3_TRAINALL<- rbind(PACKCL1_train3,PACKCL2_train3,PACKCL3_train3)
PACK3_TESTALL<- rbind(PACKCL1_test3,PACKCL2_test3,PACKCL3_test3)

# PACK 4
# CL1
packcl1p4 = PACK4[which(PACK4$SETROW=="SETROW1"),]
n <- nrow(packcl1p4)
PACKCL1p4 <- sample(1:n, 0.8*n)
PACKCL1_train4 <- packcl1p4[PACKCL1p4,]
PACKCL1_test4 <- packcl1p4[-PACKCL1p4,]

# CL2
packcl2p4 = PACK4[which(PACK4$SETROW=="SETROW2"),]
n <- nrow(packcl2p4)
PACKCL2p4 <- sample(1:n, 0.8*n)
PACKCL2_train4 <- packcl2p4[PACKCL2p4,]
PACKCL2_test4 <- packcl2p4[-PACKCL2p4,]

# CL3
packcl3p4 = PACK4[which(PACK4$SETROW=="SETROW3"),]
n <- nrow(packcl3p4)
PACKCL3p4 <- sample(1:n, 0.8*n)
PACKCL3_train4 <- packcl3p4[PACKCL3p4,]
PACKCL3_test4 <- packcl3p4[-PACKCL3p4,]

# UNIONIZING PACK 4 CLs
PACK4_TRAINALL<- rbind(PACKCL1_train4,PACKCL2_train4,PACKCL3_train4)
PACK4_TESTALL<- rbind(PACKCL1_test4,PACKCL2_test4,PACKCL3_test4)

# Test and train labels for PACK 1-4
PACK1_TRAINALL_no<- PACK1_TRAINALL[,-c(101,102)]
PACK1_TRAINALL_LABEL <- PACK1_TRAINALL[,"CL"]
PACK2_TRAINALL_no<- PACK2_TRAINALL[,-c(101,102)]
PACK2_TRAINALL_LABEL <- PACK2_TRAINALL[,"CL"]
PACK3_TRAINALL_no<- PACK3_TRAINALL[,-c(101,102)]
PACK3_TRAINALL_LABEL <- PACK3_TRAINALL[,"CL"]
PACK4_TRAINALL_no<- PACK4_TRAINALL[,-c(101,102)]
PACK4_TRAINALL_LABEL <- PACK4_TRAINALL[,"CL"]

PACK1_TESTALL_no<- PACK1_TESTALL[,-c(101,102)]
PACK1_TESTALL_LABEL <- PACK1_TESTALL[,"CL"]
PACK2_TESTALL_no<- PACK2_TESTALL[,-c(101,102)]
PACK2_TESTALL_LABEL <- PACK2_TESTALL[,"CL"]
PACK3_TESTALL_no<- PACK3_TESTALL[,-c(101,102)]
PACK3_TESTALL_LABEL <- PACK3_TESTALL[,"CL"]
PACK4_TESTALL_no<- PACK4_TESTALL[,-c(101,102)]
PACK4_TESTALL_LABEL <- PACK4_TESTALL[,"CL"]

############################ KNN After subsetting PACKS 1-4 ############################                        
                            
# Apply KNN using K = 5 to all 4 pack test sets
set.seed(1)
knn.predPACK1 <- knn(train=PACK1_TRAINALL_no,
                     test=PACK1_TESTALL_no,
                     cl = PACK1_TRAINALL_LABEL,
                     k=5)
set.seed(1)
knn.predPACK2 <- knn(train=PACK2_TRAINALL_no,
                     test=PACK2_TESTALL_no,
                     cl = PACK2_TRAINALL_LABEL,
                     k=5)
set.seed(1)
knn.predPACK3 <- knn(train=PACK3_TRAINALL_no,
                     test=PACK3_TESTALL_no,
                     cl = PACK3_TRAINALL_LABEL,
                     k=5)
set.seed(1)
knn.predPACK4 <- knn(train=PACK4_TRAINALL_no,
                     test=PACK4_TESTALL_no,
                     cl = PACK4_TRAINALL_LABEL,
                     k=5)
                            
# Find accuracy and set it to weight
w1 <- mean(knn.predPACK1 == PACK1_TESTALL_LABEL)
w2 <- mean(knn.predPACK2 == PACK2_TESTALL_LABEL)
w3 <- mean(knn.predPACK3 == PACK3_TESTALL_LABEL)
w4 <- mean(knn.predPACK4 == PACK4_TESTALL_LABEL)
# Displaying values of accuracy 
w1 #0.5486111
w2 #0.7569444
w3 #0.6319444
w4 #0.6666667
## We can see pack2 had the highest accuracy here

# Multiply weights to each pack
wpack1 <- PACK1_no*w1
wpack2 <- PACK2_no*w2
wpack3 <- PACK3_no*w3
wpack4 <- PACK4_no*w4
                            
# Bind the weighted packs together
wpackfull <- cbind(wpack1,wpack2,wpack3,wpack4,PACK1[,101])
# Normalizing the full weight packs 
Swpackfull <- wpackfull %>% mutate_if(is.numeric, function (x) as.vector(scale(x)))
                                     
# Train and test labels
waptrainset_no <- waptrainset[,-401]
waptrainset_label <- waptrainset[,401]
                                      
waptestset_no <- waptestset[,-401]
waptestset_label <- waptestset[,401]
    
#################################### Weighted KNN #################################                                       
                                      
# Global Performance for both train and test set with weighted values where knn= 5
set.seed(1)                            
knn.predwtrain <- knn(train=waptrainset_no,
                      test=waptrainset_no,
                      cl = waptrainset_label,
                      k=5)
full.predwtrain <- mean(knn.predwtrain == waptrainset_label)
full.predwtrain # 0.78 accuracy
                                     
knn.predwtest <- knn(train=waptrainset_no,
                     test=waptestset_no,
                     cl = waptrainset_label,
                     k=5)
full.predwtest<- mean(knn.predwtest == waptestset_label)
full.predwtest
                                     
# Confusion matrix
table(data.frame(knn.predwtrain,waptrainset_label))
table(data.frame(knn.predwtest, waptestset_label))
                                     
