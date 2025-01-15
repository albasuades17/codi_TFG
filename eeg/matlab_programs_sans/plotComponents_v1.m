
% block 5 requires special upload for EEGs
clear all

% Motivacions de cada bloc (de 1 a 10).
nBlockToMotivation = [ 0 0 1 1 2 2 1 1 2 2 ];

% idSubject = 33;
% subjectName = 'AP';
% subjectNickName = 'AP';
% sessionNumber = 1;
% date = {'01082017','02082017'};
% listBlocks = {[1 3 5 4 2 6],[8 2 10 9 1 7]};

% idSubject = 25;
% subjectName = 'AI';
% subjectNickName = 'AI';
% sessionNumber = 1;
% date = {'13032017','14032017'};
% listBlocks = {[1 2 8 3 5 4], [6 7 2 10 1 9]};
% nSources = 30;

%idSubject = 26;
% subjectName = 'GP';
% subjectNickName = 'GP';
% date = {'20032017','21032017'};
% listBlocks = {[1 2 10 3], [1 2 4 5 8 9]};

% idSubject = 27;
% subjectName = 'RB';
% subjectNickName = 'RB';
% date = {'22032017','23032017'};
% listBlocks = {[1 2 10 6 7 3], [5 2 4 1 8 9]};

% idSubject = 28;
% subjectName = 'TG';
% subjectNickName = 'TG';
% date = {'24032017','25032017'};
% listBlocks = {[1 2 9 7 4 6], [5 3 2 8 1 10]};

% idSubject = 29;
% subjectName = 'XP';% subjectNickName = 'XP';
% date = {'27032017', '28032017'};
% listBlocks = {[1 2 8 5], [3 6 2 10 9 1]};


% idSubject = 30;
% subjectName = 'FR';
% subjectNickName = 'FR';
% date = {'27032017', '28032017'};
% listBlocks = {[1 2 10 8 7 3], [5 6 2 9 4 1]};


% idSubject = 31;
% subjectName = 'GC';
% subjectNickName = 'GC';
% date = {'30032017', '05042017'};
% listBlocks = {[1 3 2 8], [4 5 1 6 2 7]};

% idSubject = 32;
% subjectName = 'CM';
% subjectNickName = 'CM';
% date = {'03042017', '04042017'};
% listBlocks = {[1 2 5 9 8 10], [2 4 6 1 7 3]};

% idSubject = 33;
% subjectName = 'AP';
% subjectNickName = 'AP';
% date = {'01082017','02082017'};
% listBlocks = {[1 3 5 4 2 6], [8 2 10 9 1 7]};

% idSubject = 34;
% subjectName = 'MC';
% subjectNickName = 'MC';
% date = {'01082017','02082017'};
% listBlocks = {[2 6 4 3 1 5], [8 2 10 9 1 7]};

% idSubject = 35;
% subjectName = 'MM';
% subjectNickName = 'MM';
% date = {'25092017','27092017'};
% listBlocks = {[1 3 5 4 2 6], [8 10 2 9 1 7]};


% Directori on hi ha les dades dels pacients sans
basicDir = 'C:\Users\albas\OneDrive\Desktop\TFG\TFG_DB\eeg\pacients_sans\EEG-BCN\';
DSFactor = 2;             
                                      
%cd fieldtrip
ft_defaults
                                      
% list of blocks to load
nTrialsBlock = 108;

% lBlocks1 = []; lBlocks2 = [];
% Duració de cada assaig
nlength = 1200;
dataM = nan(60,nlength,108);


for idSubject=25:34,%25:34,

    for px=1, % interval 1, interval 2
        % ic_data2 conté les dades de la fonts del pacient idSubject
        load(sprintf('%s/dataClean-ICA3-%d-T%d.mat',basicDir, idSubject,px), ...
            'ic_data2');
        load(sprintf('%s/dataClean-ICA3-%d-T%d.mat',basicDir, idSubject,px), ...
            'ic_data3');

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
        
        cfgic = [];
        % Vector que conté cada una de les fonts
        cfgic.component          =  1:nSources;
        cfgic.layout             =  'ActiCap64_LM.lay';
%         g1 = figure;
%         ft_topoplotIC(cfgic, ic_data2);
%         set(g1,'color',[1 1 1]);
% 
%         set(gcf, 'Position', get(0, 'Screensize'));
%         print(g1,'-dpdf','-r300','-bestfit',sprintf('%s/cleanData18092019/Sources-%d-T%d.pdf',basicDir,idSubject,px));
        saveDir = 'C:\Users\albas\OneDrive\Desktop\TFG\TFG_DB\eeg\pacients_sans\sources\'
        % plot surces and extract source positions
        loc = getSourceCoefficientOnElectrodes(cfgic, ic_data2, idSubject, saveDir);

        cfgt = [];
        cfgt.viewmode = 'component';
        %cfgt.continuous = 'yes';
        %cfgt.blocksize = 30;
        cfgt.channels = [1:nSources];
        cfgt.layout             =  'ActiCap64_LM.lay';
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