x = load('ex3x.dat');
y = load('ex3y.dat');

m = length(y);


x = [ones(m, 1), x];

x_unscaled = x;

mu = mean(x)

sigma = std(x);

x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

figure
plotyle ={'b', 'r', 'g', 'k', 'b--', 'r--'};

alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3];

MAX_ITR = 100;

theta_grad_decent = zeros(size(x(1,:)))

for i = 1:length(alpha)
    theta = zeros(size(x(1,:)))';
    J = zeros(MAX_ITR, 1);
    for num_iterations = 1:MAX_ITR
        J(num_iterations) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
        
        grad = (1/m) .* x' * ((x * theta) - y);

        theta = theta - alpha(i) .* grad;
    end
    plot(0:49, J(1:50), char(plotstyle(i)), 'LineWidth', 2)
    hold on

end

legend('0.01', '0.03', '0.1', '0.3', '1', '1.3')
xlabel('Number of Iterations');
ylabel('Cost J')
format long