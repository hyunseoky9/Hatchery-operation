getwd()
generate_data <- function(samples = 1000000) {
  # Generate random inputs for the 6 variables
  x1 <- runif(samples, -pi, pi)    # Uniform random between -π and π
  x2 <- runif(samples, -pi, pi)    # Uniform random between -π and π
  x3 <- runif(samples, 1, 10)      # Uniform random between 1 and 10
  x4 <- runif(samples, 1, 10)      # Uniform random between 1 and 10
  x5 <- runif(samples, -1, 1)      # Uniform random between -1 and 1
  x6 <- runif(samples, -1, 1)      # Uniform random between -1 and 1
  
  a = rbinom(samples, 1, 0.5)
  # Compute the target function
  # zero vector with length of samples
  y = rep(0, samples)
  idx1 = which(a == 1)
  idx2 = which(a == 2)
  y1 = sin(x1) + x2^2 - log(abs(x3) + 1) + exp(x4) - x5 * x6
  y2 = x1^2 + sin(x2) + (x3 * x4) + exp(-(x5^2 + x6^2))
  
  # Combine inputs and output into a data frame
  dataset <- data.frame(x1, x2, x3, x4, x5, x6, y1, y2)
  return(dataset)
}

dataset <- generate_data(100000)

# Save to CSV if needed
write.csv(dataset, "generated_dataset_test.csv", row.names = FALSE)

# Display the first few rows
head(dataset)

#
library(ggplot2)
