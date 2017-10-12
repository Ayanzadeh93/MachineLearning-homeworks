#rstudio does not work correctly in ubuntu, i can not cheeck the R file in R studio, i run my program in jupyter notebook in R, please open jupyer file(HW1.ipynb).
#library(repr)       #if you want to resize the size of plot make it as code

mylist <- NULL#normal distribution
for (year in 1:500){
  univariate_random_variable<-rnorm(10,0,1)
  
    mylist[year]<- mean(univariate_random_variable)
    
}
#for 10 sample

mylist1 <- NULL
for (year in 1:500){
  univariate_random_variable_100<-rnorm(100, mean = 0,sd = 1)
  
    mylist1[year]<- mean(univariate_random_variable_100)
   
     
   
}

#for 100 SAMPLE

#options(repr.plot.width=7, repr.plot.height=5)
#par(mfrow=c(1,2))
hist(mylist1, main="Histogram of 100 sample",xlab="samples",ylab = "number of samples", col="skyblue",ylim = c(0,120),
     xlim=c(-1,1), las=1, 
     breaks=20)
#options(repr.plot.width=5, repr.plot.height=5)


hist(mylist, main="Histogram of 10 samples",xlab="samples",ylab = "number of samples", col="pink",ylim = c(0,80),
     xlim=c(-2,2), las=3, 
     breaks=20)
#legend("topright", c("sample Frequency"), col=c("blue"), lwd=10)

m1=0
st1=1
m2=1
st2=2
c1=0.50


y<-NULL
z<-NULL

y=rnorm(300000,mean=m1,st1)
z=rnorm(300000,mean = m2,st2)
#mean and standard deviation of  datasets
#Generate two random datasets from p(x|C1) and p(x|C2)
mean1=mean(y)
stdv1=sd(y)
mean2=mean(z)
stdv2=sd(z)

print(mean1)
print(mean2)
print(stdv1)
print(stdv2)

options(repr.plot.width=6, repr.plot.height=6)
hist(y, prob=TRUE, ylim=c(0,0.5), breaks=10, xlim = c(-10,10),col = 'yellow',xlab="x",ylab="p(x|ci)", main = "Likelihoods of classes" )
hist(z, prob=TRUE, ylim=c(0,0.5), breaks=10, xlim = c(-10,10),add=T, col = 'green')

g1 = function(x){ #equation of discriminant function of g1
    dnorm(x, mean = 0, sd = 1, log = FALSE)
}
lines(density(y),col ='blue',lwd = 2 )
curve(g1, from=-10, to=10,n=300
      ,  xlab="x", ylab="y", col="blue",lwd=2,add=T)

g2= function(x){#equation of discriminant function of g2
   dnorm(x, mean = 1, sd = 2, log = FALSE)}
curve(g2, from=-10, to=10, n=300, add=TRUE, xlab="X", ylab="Y", col="red",lwd=2, 
             )
lines(density(z),col ='red',lwd = 2 )
legend("topright", c("p(x/c1)", "p(x/c2)"), col=c("blue", "red"), lwd=10)

options(repr.plot.width=6, repr.plot.height=6)
g1 = function(x){ #equation of discriminant function of g1
    
    dnorm(x, mean = m1, sd = st1, log = FALSE)/ (dnorm(x, mean = m2, sd = st2, log = FALSE)+dnorm(x, mean = m1, sd = st1, log = FALSE))
}

curve(g1, from=-10, to=10,n=300, ylim=c(0,1)
      ,  xlab="x", ylab="P(c1/x)and p(c2/x)", col="blue",lwd=2,main="Plot P(c1/x)and p(c2/x) ")

g2= function(x){#equation of discriminant function of g2
    
   dnorm(x, mean = m2, sd = st2, log = FALSE)}/(dnorm(x, mean = m2, sd = st2, log = FALSE)+dnorm(x, mean = m1, sd = st1, log = FALSE))
curve(g2, from=-10, to=10, n=300, add=TRUE, xlab="X", ylab="Y", col="red",lwd=2, 
             main="Plot P(c1/x and p(c2/x)) ")

legend("topright", c("g1(x)", "g2(x)"), col=c("blue", "red"), lwd=10)
#grid (10,10, lty = 6, col = "cornsilk2")
axis(side = 2, at =0.5,labels = "0.5")

#finding the intersection points of g1 and g2
c1=0.5
c2=0.5
p=3
q=2
z=-8*(log(c1/c2)+log(2)+1/8)
# determining the decision Region.
decision_region <- function(a,b,c){
  if(delta(a,b,c) > 0){ # first case Delta>0
        x1 = (-b+sqrt(delta(a,b,c)))/(2*a)
        x2 = (-b-sqrt(delta(a,b,c)))/(2*a)
        result = c(x1,x2)
      
  }
  else if(delta(a,b,c) == 0){ # delta equals to zero
      print("one decision region")
        x = -b/(2*a)
  }
  else {print("no decession Region" )
        
       } # delta is less than zero
}

delta<-function(a,b,c){
      b^2-4*a*c
}

decision <- decision_region(p,q,z); decision


options(repr.plot.width=6, repr.plot.height=6)
g1 = function(x){ #equation of discriminant function of g1
    (((-1/2)*log(2*pi))-(log(st1))-((x-m1)^2/(2*st1^2))+log(0.5))}

curve(g1, from=-5, to=10, n=300, xlab="x", ylab="y", col="blue",lwd=2,main="discriminant function of g1(x) and g2(x) ")

g2= function(x){#equation of discriminant function of g2
    (((-1/2)*log(2*pi))-(log(st2))-((x-m2)^2/(2*(st2^2)))+log(0.5))}
                 
curve(g2, from=-5, to=10, n=300, add=TRUE, xlab="X", ylab="Y", col="red",lwd=2
             )
x4 <- c(1.18087831829851, 3, 5, 8, 12)
y4 <- c(-2.25, 1, 2, 4, 6)
points(x4, y4, pch=16, col="green")
abline(v=-1.84754498496518, col="black",lwd=1, lty=2)
abline(v=1.18087831829851, col="black",lwd=1, lty=2)
x2 <- c(-1.84754498496518, 3, 5, 8, 12)
y2 <- c(-3.369, 1, 2, 4, 6)

points(x2, y2, pch=16, col="green")

legend("topright", c("g1(x)", "g2(x)"), col=c("blue", "red"), lwd=10)
text(0,-30, "R1", col = "blue" ,lwd=1)
text(-4,-30, "R2", col = "RED" ,lwd=1)
text(5,-30, "R2", col = "red" ,lwd=1)
axis(1, at =-1.84754498496518,labels = "x1") 
axis(1, at =1.18087831829851,labels = "x2")


g1 = function(x){ #equation of discriminant function of g1
    dnorm(x, mean = 0, sd = 1, log = FALSE)
}

curve(g1, from=-10, to=10,n=300
      ,  xlab="x", ylab="y", col="blue",lwd=2)

g2= function(x){#equation of discriminant function of g2
   dnorm(x, mean = 1, sd = 2, log = FALSE)}
curve(g2, from=-10, to=10, n=300, add=TRUE, xlab="X", ylab="Y", col="red",lwd=2, 
             )

x4 <- c(1.18087831829851, 3, 5, 8, 12)
y4 <- c(-2.25, 1, 2, 4, 6)
points(x4, y4, pch=16, col="green")
abline(v=-1.84754498496518, col="black",lwd=1, lty=2)
abline(v=1.18087831829851, col="black",lwd=1, lty=2)
x2 <- c(-1.84754498496518, 3, 5, 8, 12)
y2 <- c(-3.369, 1, 2, 4, 6)

points(x2, y2, pch=16, col="green")
legend("topright", c("g1(x)", "g2(x)"), col=c("blue", "red"), lwd=10)
text(-4,0.1, "R2", col = "RED" ,lwd=1)
text(0,0.25, "R1", col = "blue" ,lwd=1)
text(5,0.1, "R2", col = "red" ,lwd=1)
#lines(density(z),col ='red',lwd = 2 )
legend("topright", c("p(x/c1)", "p(x/c2)"), col=c("blue", "red"), lwd=10)
axis(1, at =-1.84754498496518,labels = "x1") 
axis(1, at =1.18087831829851,labels = "x2")

g1 = function(x){ #equation of discriminant function of g1 and g2 when p(c2) increase to 0.8 
    (((-1/2)*log(2*pi))-(log(st1))-((x-m1)^2/(2*st1^2))+log(0.2))}

curve(g1, from=-10, to=10, n=300, xlab="x", ylab="y", col="blue",lwd=2,main="discriminant function of g1(x) and g2(x) p(c2)=0.8 ")

g2= function(x){#equation of discriminant function of g2
    (((-1/2)*log(2*pi))-(log(st2))-((x-m2)^2/(2*(st2^2)))+log(0.8))}
                 
curve(g2, from=-10, to=10, n=300, add=TRUE, xlab="X", ylab="Y", col="red",lwd=2
             )
legend("topleft", c("g1(x)", "g2(x)"), col=c("blue", "red"), lwd=10)
#grid (10,10, lty = 6, col = "cornsilk2")
text(0,-40, "R2", col = "red" ,lwd=1)
