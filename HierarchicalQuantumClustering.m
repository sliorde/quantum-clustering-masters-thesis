function cluster_hierarchy = HierarchicalQuantumClustering(data,sigma)
% Perform Hierarchical Quantum Clustering (QC).
%
% Arguments:
%    'data':
%       An observation matrix, where each row is an observation and each 
%       column is a feature.
%    'sigma': 
%       A column vector, with strickly increasing values. HQC finds 
%       clusters for each scale defined by these values.
%
% Returns:
%    'cluster_hierarchy': 
%       A matrix with the multiscale clusterings.
%       'cluster_hierarchy(ii,jj)' is the cluster assignment of 
%       'data(ii,:)' in scale 'sigma(jj)'. Cluster designations are 
%       numbers, ordered decreasingly by the cluster sizes.
	
	x = data;
	inds1 = 1:size(data,1);
	cluster_hierarchy = zeros(size(data,1),numel(sigma));
	for ii=1:numel(sigma)
		display(['step ' num2str(ii) '/' num2str(numel(sigma))]);
		display(['number of clusters: ' num2str(size(x,1))]);
		display(' ');
		
		% Perform QC with current value of 'sigma', on representative 
		% points x.
		clusters = QuantumClustering(data,sigma(ii),x);
		
		% 'clusters' contains clusters only for representative points.
		%  Distribure them among original data points.
		clusters = clusters(inds1);
		
		% Change cluster labels so that they are ordered by size (largest
		% cluster is 1, second largest is 2, etc.)
		[~,inds2] = sort(accumarray(clusters,1),'descend');
		[~,inds2] = sort(inds2);
		clusters = inds2(clusters);
	
		cluster_hierarchy(:,ii) = clusters;

		% Get one representative point for each cluster
		[~,inds2,inds1] = unique(clusters);
		x = data(inds2,:);
	end
end