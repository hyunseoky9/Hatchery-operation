samplenum = 1000;

n = 1000;
p = 0.9;

% generate using binornd
samples = binornd(n, p, samplenum, 1);


% generate using normal distribution 
mu = n*p;
sigma = sqrt(n*p*(1-p));
samples2 = normrnd(mu, sigma, samplenum, 1);

% plot
figure(1);
clf;
subplot(2,1,1);
histogram(samples, 100);
xlabel('Value');
ylabel('Frequency');
title('Binomial distribution');

subplot(2,1,2);
histogram(samples2, 100);
xlabel('Value');
ylabel('Frequency');
title('Normal distribution');
