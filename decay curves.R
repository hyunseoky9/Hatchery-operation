episodenum = 2000
x = seq(0,episodenum,by=1)

# logistic
fix = 2000
a=0.15
b=-30/fix
c=-fix*0.6
c
d=0
y = a/(1+exp(-b*(x+c))) 
plot(y~x,type='l')
#which(y)

# inverse
#a = 1
#y2 = 1/(a*x)
#lines(y2~x,col='red')

# exponential
#a = 0.1
#b = 17/episodenum
#y3 = a*exp(-b*x)
#plot(y3~x,col='blue')

# exponential a (the one used in DQN and DRQN)
#a = 0.9995
#lrstart = 0.01
#y4 = lrstart*a^(x)
#plot(y4~x,type='l')
#y4[(length(y4)-100):length(y4)]
#