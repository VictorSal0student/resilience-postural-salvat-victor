function phivec_interp_sampled=relative_phase_from_discrete_events(time,timestamps_probe,timestamps_target)
%{
Takes in a time vector, (i.e. 0:dt:lengthoftrial) and two
vectors of time stamps (that should fit within the limits of the time). The
second timestamps vector is interpolated linearly in phase and then
"probed" by the first timestamps vector. For example,
t=0:.1:10;x=[1:9];y=[1:.5:9];
z1=relative_phase_continuous_from_discrete_events(t,x,y);
z2=relative_phase_continuous_from_discrete_events(t,y,x);
%}

% Make a binary vector indicating the points where the probe happens. Same size as time
timestamps_probe_bin = time_stamps_convert_to_bin(time,timestamps_probe);

% Make an interpolated continuous relative phase time series for the target.
phivec_interp = phase_lin_interpolated(time,timestamps_target);

% Sample the target interpolated time series using the probe binary.
phivec_interp_sampled = mod(phivec_interp(timestamps_probe_bin),1*2*pi);

% Discard nans.
phivec_interp_sampled = phivec_interp_sampled(~isnan(phivec_interp_sampled));

%% Function to make a binary vector from time stamps
    function timestampsbin=time_stamps_convert_to_bin(timevec,timestamps)
        if timestamps(end)>timevec(end);disp('Warning! Time stamps run out of time!');end
        timestampsbin=zeros(size(timevec));
        for k=1:length(timestamps)
            [~,beat_index]=min((timevec-timestamps(k)).^2);
            timestampsbin(beat_index)=1;
        end
        timestampsbin=logical(timestampsbin);
    end

%% Function to make interpolated continuous relative phase time series
    function phivecinterp=phase_lin_interpolated(timevec,stepstimevec)
        kstart=find(timevec>=stepstimevec(1),1,'first');
        kend  =find(timevec<=stepstimevec(end),1,'last');
        phivecinterp=timevec.*nan;
        for k=kstart+1:kend
            % Find step time just before time k.
            tk=stepstimevec(find(stepstimevec<timevec(k),1,'last'));
            % Find step time just after time k.
            tk1=stepstimevec(find(stepstimevec>=timevec(k),1,'first'));
            % Eq. 1.295 in Anishchenko et al. 2007 book , *2.
            phivecinterp(k)=2*pi*((timevec(k)-tk)/(tk1-tk)+sum(stepstimevec<timevec(k))-1);
        end
    end

end