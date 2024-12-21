episodenum = 20000
x = seq(0,episodenum,by=1)

# logistic
#fix = 40000
#a=1
#b=-10*1/fix*3
#c=-fix*0.3
#c
#d=0
#y = a/(1+exp(-b*(x+c))) 
#plot(y~x,type='l')
#which(y)

# inverse
#a = 1
#y2 = 1/(a*x)
#lines(y2~x,col='red')

# exponential
#a = 0.1
#b = 1/episodenum*13
#y3 = a*exp(-b*x)
#which(y3<0.001)[1]
#lines(y3~x,col='blue')

# exponential a
a = 0.9995
lrstart = 0.01
y4 = lrstart*a^(x)
plot(y4~x,type='l')
y4[(length(y4)-100):length(y4)]
