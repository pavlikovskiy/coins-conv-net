
function [csvdata_reduced] = filterDataset(csvdata, max_class_samples)
%%
% make distribution training samples over labels more smooth
% filter out samples if amount more than max_class_samples

labels = unique(csvdata(:, 2))';

%sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
y = csvdata(:, 2); % second column is coinIdx

csvdata_reduced = zeros(0, 0);

y_sparse = repmat(y, 1, size(labels, 2));
l_sparse = repmat(labels, size(y, 1), 1);
bin = (y_sparse == l_sparse); % sparse matrix

for i = 1:max(labels)
    ind = find(bin(:, i));
    
    max_samples = max_class_samples;
    if max_class_samples > size(ind, 1)
        max_samples = size(ind, 1);
    end
    
    
    csvdata_reduced = [csvdata_reduced; csvdata(ind(1:max_samples), :)];    
    
    if max_class_samples > size(ind, 1)        
        max_extra = max_class_samples - size(ind, 1);
        if max_extra > size(ind, 1)
            max_extra = size(ind, 1);
        end
        
        csvdata_reduced = [csvdata_reduced; csvdata(ind(1:max_extra), :)];    
    end
end


end
