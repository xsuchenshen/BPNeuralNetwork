function result = getWeightChange(filename, hiddenLayer, nodeSample)
%     coordinates = 3;
    data = dlmread(filename);
    dataSize = size(data);
    columns = hiddenLayer * nodeSample;
    rows = dataSize(1) / columns - 1;
    result = zeros(rows, columns);
    for layer=1:hiddenLayer
      for node=0:nodeSample-1
        subdata = data(data(:,2)==layer & data(:,3)==node,4:end);
        angles = zeros(rows-1,1);
        for row=2:rows+1
          dotproduct = dot(subdata(row,:), subdata(row-1,:));
          normproduct = norm(subdata(row,:)) * norm(subdata(row-1,:));
          angles(row-1, 1) = acos(dotproduct/normproduct);
        end
        result(:,(layer-1)*hiddenLayer+node+1) = angles;
      end
    end
end