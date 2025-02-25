# modeled probability distribution of RGSM catch (CPUE) (negative binomial) with different parameters
Ns = c(1000, 3000, 15195, 76961, 389806, 1974350, 10000000)
p = 0.05
f = 0.01
r = 5
x = seq(0, 60, 1)

# Define colors for each plot
colors = rainbow(length(Ns))

# Plot setup
plot(NULL, xlim=c(0, 60), ylim=c(0, 0.61), xlab="CPUE",main=sprintf('catch probability distritbution (detection=%.2f, dispersion=%.2f)',p,r), ylab="Probability", type='n')

# Loop through each N in Ns and plot the distribution
for (i in 1:length(Ns)) {
    N = Ns[i]
    C = N * p * f
    y = dnbinom(x, size=r, mu=C, log = FALSE)
    lines(x, y, col=colors[i], type='l', lwd=3)  # Increase line width with lwd parameter
}

# Add legend
legend("topright", legend=paste("N =", Ns), col=colors, lty=1)

print(C)
sample = rnbinom(1, size=r, mu=C)
print(sample)
