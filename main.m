system('make clean');
system('make');
system('./all');
filename='../data/WEIGHTCHANGE.txt';
hiddenLayer=2;
nodeSample=2;
result = getWeightChange(filename, hiddenLayer, nodeSample);

% figure 1: Nodes in the same layer
figure(1);
for i=0:hiddenLayer-1
    subplot(hiddenLayer, 1, i+1);
    plot(result(:,i*nodeSample+1:i*nodeSample+nodeSample));
    legends = cell(nodeSample, 1);
    for j=1:nodeSample
        legends{j} = sprintf('Node %d', j);
    end
    legend( legends );
    xlabel('Iteration');
    ylabel('Weight change');
    title(sprintf('Weight change for the first %d nodes in hidden layer %d', nodeSample, i));
end

% figure 2: Nodes in different layers
figure(2);
for i=0:nodeSample-1
    subplot(nodeSample, 1, i+1);
    columnVector = zeros(1, hiddenLayer);
    legends = cell(hiddenLayer, 1);
    for j=1:hiddenLayer
        columnVector(1, j) = (j - 1) * nodeSample + i + 1;
        legends{j} = sprintf('Layer %d', j);
    end
    plot(result(:,columnVector));
    legend( legends );
    xlabel('Iteration');
    ylabel('Weight change');
    title(sprintf('Weight change for node %d in all hidden layers', i));
end

% figure 3:
figure(3);
line([1:size(result,1)], result(:,2), 'Color', 'r');
xlabel('Iteration');
ylabel('Weight change');
ax1 = gca;
set(ax1,'XColor','r','YColor','r');
legend('Weight change of one node', 'Location', 'northwest');
data_error=dlmread('../data/ERRORCHANGE.txt');
ax2 = axes('Position',get(ax1,'Position'),... 
'XAxisLocation','top',... 
'YAxisLocation','right',... 
'Color','none',... 
'XColor','b','YColor','b'); 
line([1:size(result,1)+1], data_error(:,1), 'Parent', ax2, 'Color', 'b');
line([1:size(result,1)+1], data_error(:,2), 'Parent', ax2, 'Color', 'k');
xlabel(ax2, 'Iteration');
ylabel(ax2, 'Error');
h = legend(ax2, 'Test error', 'Training error', 'Location', 'northeast');
set(h, 'color', 'w');

data_error=dlmread('../data/ERRORCHANGE.txt');
figure(4);
plot(data_error);
legend('Test Error','Training Error')
title('Error Changes over Learning Cycles');

data_activation=dlmread('../data/ACTIVATIONCHANGE.txt');
sz_activation= size(data_activation);
number=(sz_activation(2)-1)/3;
iteration=data_activation(1,1);
figure(5);
for i = 1:number
    layer_num=data_activation(1,(i-1)*3+2);
    node_num=data_activation(1,(i-1)*3+3);
    
    subplot(1,number,i);
    hist(data_activation(:,(i-1)*3+4),20);
    s = strcat('Node ',num2str(node_num),' of ',num2str(layer_num),'th Hidden Layer(Iteration ',num2str(iteration),')');
    title(s);
end
