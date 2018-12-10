clear all;
close all;

n = 130;
m = 4;
x = ones(n+1:1);
y = ones(n:1);

for i = 1:n+1
    x(i) = prod(i:n)/factorial(n+1-i)/1000000; 
end

for i = 1:n
    y(i) = x(i)*x(i+1);
end
y_sum = sum(y);
x_sum = sum(x);

temp = 0;
result = ones(m:1);
j = 1;

for i = 1:n-1
   temp = temp + y(i);
    if(temp+0.5*y(i+1) >= j*y_sum/m) 
        result(j) = i;
        j = j+1;
   end
end
result(m) = n;
plot(y);