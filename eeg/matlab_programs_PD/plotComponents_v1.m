
% block 5 requires special upload for EEGs
clear all

% Motivacions de cada bloc (de 1 a 10).
nBlockToMotivation = [ 0 0 1 1 2 2 1 1 2 2 ];

% Directori on hi ha les dades dels pacients de Parkinson
basicDir = 'C:\Users\albas\OneDrive\Desktop\TFG\TFG_DB\eeg\pacients_PD\PD-EEG\';
DSFactor = 2;             
                                      
%cd fieldtrip
ft_defaults
                                      
% list of blocks to load
nTrialsBlock = 108;

% lBlocks1 = []; lBlocks2 = [];
% Duració de cada assaig
nlength = 1600;
dataM = nan(60,nlength,108);

% Agafem els pacients que tenen ic_data2
for idSubject=[55,60,61,64],

    for px=1, % interval 1, interval 2
        % ic_data2 conté les dades de la fonts del pacient idSubject
        load(sprintf('%s/dataClean-ICA-%d-T%d.mat',basicDir, idSubject,px), ...
            'ic_data2');
        load(sprintf('%s/dataClean-ICA-%d-T%d.mat',basicDir, idSubject,px), ...
            'ic_data');

        ic_data3 = ic_data;

        % plot sources
        ic_data4 = ic_data2;
        % Quantitat de fons (del ICA)
        nSources = size(ic_data2.label,1);
        
        for nB = size(ic_data3,4),
            for nT=1:size(ic_data3,3),
               % S'agafen les dades de cada trial i de cada bloc i es posen
               % en una sola matriu (s'aplanen les dades)
               % És a dir, tindrem una dimensió de 108*12
               ic_data4.trial{  (nB-1)*108+nT  } = ic_data3(:,:,nT,nB);
            end
        end
        ic_data2 = ic_data4;

        % Veiem si ic_data2.time i ic_data2.trial tenen la mateixa length
        len_time = length(ic_data2.time);
        len_trial = length(ic_data2.trial);
        
        if len_time ~= len_trial
            % Agafem la length menor.
            min_len = min(len_time, len_trial);
            ic_data2.time = ic_data2.time(1:min_len);
            ic_data2.trial = ic_data2.trial(1:min_len);
        end

        n_labels = length(ic_data2.label);

        % We trim de trials and labels so the size matches
        for i = 1:length(ic_data2.trial)
            ic_data2.trial{i} = ic_data2.trial{i}(1:n_labels, :);
        end

        cfgic = [];
        % Vector que conté cada una de les fonts
        cfgic.component          =  1:nSources;
        cfgic.layout             =  'GSN129.sfp';
%         g1 = figure;
%         ft_topoplotIC(cfgic, ic_data2);
%         set(g1,'color',[1 1 1]);
% 
%         set(gcf, 'Position', get(0, 'Screensize'));
%         print(g1,'-dpdf','-r300','-bestfit',sprintf('%s/cleanData18092019/Sources-%d-T%d.pdf',basicDir,idSubject,px));
        saveDir = 'C:\Users\albas\OneDrive\Desktop\TFG\TFG_DB\eeg\pacients_PD\sources\';
        % plot surces and extract source positions
        loc = getSourceCoefficientOnElectrodes(cfgic, ic_data2, idSubject, saveDir);

        cfgt = [];
        cfgt.viewmode = 'component';
        %cfgt.continuous = 'yes';
        %cfgt.blocksize = 30;
        cfgt.channels = [1:nSources];
        cfgt.layout             =  'GSN129.sfp';
        ft_databrowser(cfgt,ic_data2);  

    %         ft_databrowser
    %         % 
    %         cfgic.viewmode = 'component';
    %         cfgic2 = ft_databrowser(cfgic, ic_data)


%         save(sprintf('%s/cleanData18092019/dataClean-ICA-%d-T%d.mat',basicDir, idSubject,px), ...
%                 'ic_data','listM','listD','ic_data2','dataSorted','dataAvgSorted','dataSortedDS','blockOrder','iMotivSortedBlocks');


    end
    clear('ic_data2','ic_data3','ic_data4');
    
    close all;
    
end