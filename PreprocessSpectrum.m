function [w_out,r_out] = PreprocessSpectrum(w,r)
% Preprocess a given spectrum to be used in the clustering algorithm.
% Returns the preprocessed spectrum, or an empty if preprocessing 
% conditions do not hold.
%
% Arguments:
%    'w':
%       A column vector with the wavelengths, in units um (micrometer).
%    'r':
%       A column vector of the same size as 'w', with reflectances.
%
% Returns:
%    'w_out': 
%       A column vector with the wavelengths 0.5um to 2.43um, in 0.02um
%       steps.
%    'r_out': 
%       A column vector the same size as 'w_out' with the reflecatances
%       after preprocessing. It is obtained by smoothing 'r' and
%       interpolating it linearly to the wavelengths in 'w_out'.
%
% If the range of wavelengths does not include the range 0.5um to 2.43um, 
% or there is a gap larger than 0.1um in that range, the spectrum cannot be
% processed and the output is empty vectors.

	start_wavelength = 0.5; % um
	stop_wavelength = 2.43; % um
	step_wavelength = 0.02; % um

	max_wavelength_gap = 0.1; % um

	
	[w,inds] = sort(w);
	r = r(inds);

	% remove reflectances that clearly indicate an error in measurement
	w = w(r>0.2);
	r = r(r>0.2);
	if numel(w) < 3
		w_out = [];
		r_out = [];
		return;
	end
	
	% if the interpolation range is not contained in the spectrum, can't
	% proceed
	if (w(1) > start_wavelength) || (w(end) < stop_wavelength)
		w_out = [];
		r_out = [];
		return;
	end
	
	% for wavelengths with multiple values, take the average
	[w,~,inds] = unique(w);
	r = accumarray(inds,r)./accumarray(inds,1);
	
	% if there is a wavelength gap too big, can't proceed
	largest_gap = max(diff(w((w>=start_wavelength) & (w<=stop_wavelength))));
	if (~isempty(largest_gap) && (largest_gap>max_wavelength_gap))
		w_out = [];
		r_out = [];
		return;
	end
	
	% smooth prior to interpolation, using a quadratic kernel with width of
	% the step size of interpolation
	r_smoothed = r;
	for ii=1:numel(w)
		inds = find(abs(w-w(ii))<step_wavelength/2);
		weights = (step_wavelength/2-abs(w(inds)-w(ii))).^2;
		weights = weights/sum(weights);
		r_smoothed(ii) = sum(r(inds).*weights);
	end
	r = r_smoothed;
	
	% interpolate
	w_out = [start_wavelength:step_wavelength:stop_wavelength];
	r_out = interp1(w,r,w_out,'linear');

end