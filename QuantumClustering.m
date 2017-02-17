function [clusters,x] = QuantumClustering(data,sigma,x)
% Perform Quantum Clustering (QC).
%
% Arguments:
%    'data':
%       An observation matrix, where each row is an observation and each 
%       column is a feature.
%    'sigma':
%       A scalar. QC finds clusters at the scale defined by this value.
%    'x':
%       (optional) A matrix with inital locations of replica points. It is
%       a matrix where each row is a replica and each column a feature. 'x'
%       does not need to have the same number of rows as 'data', since 
%       sometimes it is of interest to cluster points other then the ones 
%       in 'data'. If 'x' is not specified, or if an empty matrix is used, 
%       the default initialization 'x=data' will be used.
%
% Returns:
%    'clusters':
%       A column vector with cluster assignments for each replica point in
%       'x' (if 'x' is not specified, then 'x=data'). 'clusters(ii)' is the
%       cluster assignment of 'x(ii,:)'. Cluster designations are numbers, 
%       ordered decreasingly by the cluster sizes.
%    'x':
%       The locations of the replica points, after QC dynamics has
%       conveged.

	
	if ~exist('x','var') || isempty(x)
		x = data;
	end
	
	optimizer_parameters = optimoptions('fminunc','Algorithm',...
		'quasi-newton','GradObj','on','Display','off');
	target_function = @(x)V(data,sigma,x);
	
	for ii=1:size(x,1)
		y = fminunc(target_function,x(ii,:),optimizer_parameters);
		x(ii,:) = y;
	end
	
	clusters = PerformFinalClustering(x,sigma/10);

end

function [v,dv] = V(data,sigma,x)
% Calculate potential and its gradient.
%
% Arguments:
%   'data':
%      An observation matrix, where each row is an observation and each 
%      column is a feature. It is used to generate the potential.
%   'sigma':
%      A scalar. The width of the Gaussian kernels.
%   'x':
%      A matrix of points where the potential is to be evaluated. Each row
%      is a point and each column is a feature.
%
% Returns:
%    'v':
%       A vector with the same number of rows as 'x', with the potential 
%       values for corresponding 'x' points.
%    'dv':
%       A matrix with the same size as 'x', with the gradient in each row
%       for corresponding 'x' points.

	q = (1/(2*sigma^2));
	differences = bsxfun(@minus,x,data);
	squared_differences = sum(differences.^2,2);
	gaussians = exp(-q*squared_differences);
	one_over_parzen = 1/sum(gaussians);
	probabilities = gaussians*one_over_parzen;
	v = q*sum(squared_differences.*probabilities);
	
	dv = 2*q*sum(bsxfun(@times,differences,...
		(1+v-q*squared_differences).*probabilities));
end

function clusters = PerformFinalClustering(x,dr)
% Perform final clustering of data after the quantum clustering dynamics
% has converged.
% 
% Arguments:
%    'x':
%       Locations of replica points, after QC dynamics has converged. It is
%       a matrix where each row is a replica and each column a feature.
%    'dr':
%       A scalar. Replica points will be grouped together into a cluster if
%       they all lie within a distance smaller then 'dr' from eachother.
%
% Returns:
%    'clusters':
%       A column vector with cluster assignments for each replica point.
%       'clusters(ii)' is the cluster assignment of 'x(ii,:)'. Cluster
%       designations are numbers, ordered decreasingly by the cluster 
%       sizes.

	if size(x,1)==1
		clusters = 1;
		return;
	end

	clusters = zeros(size(x,1),1);
	ii = 1;
	c = 1; % The designation of the next cluster
	distances = squareform(pdist(x));
	while ~isempty(ii)
		inds = find(clusters==0);
		clusters(inds(distances(ii,inds) <= dr))= c;
		c= c+1;		
		ii = find(clusters==0,1,'first');
	end
	
	% Give new cluster designations, sorted by cluster sizes.
	[~,inds] = sort(accumarray(clusters,1),'descend');
	[~,inds] = sort(inds);
	clusters = inds(clusters);
end