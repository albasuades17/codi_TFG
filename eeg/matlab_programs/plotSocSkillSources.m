clear all;

g2 = figure; 
cmap = jet;
cmap(end,:) = [1,1,1];
colormap(cmap);

nFig = 6;
listF = {'alpha','beta','gamma'};
basicDir = 'C:/Users/albas/OneDrive/Desktop/TFG/TFG_DB/eeg/';
variables = {'SF', 'SP'};
folders = {'pacients_PD/common_src_%s_OFF/','pacients_PD/common_src_%s_ON/', 'pacients_sans/common_src_%s/', 'common_src_mean_%s/common_src_%s_ON_OFF/', 'common_src_mean_%s/common_src_%s_sans_OFF/', 'common_src_mean_%s/common_src_%s_sans_ON/'};
save_files = {'/%s_OFF_ST_socskill.pdf','/%s_ON_ST_socskill.pdf','/%s_sans_ST_socskill.pdf','/%s_mean_ON_OFF_ST_socskill.pdf', '/%s_mean_sans_OFF_ST_socskill.pdf','/%s_mean_sans_ON_ST_socskill.pdf'};
script = 'mean_best_src_%s_gp%d.mat';
outputDir = fullfile(basicDir, 'images_brain');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

for v=1:numel(variables)
    variable = variables{v};
    for f=6:numel(folders)
        folder = strcat(basicDir,folders{f});
        folder = strcat(folder,script);
        for m=1:3
            for k=1:6
                try
                if (f > 3)
                    load(sprintf(folder,variable,variable,listF{m},k-1));
                else
                    load(sprintf(folder,variable,listF{m},k-1));
                end
        
                i1 = find(src==0); 
                src(i1)=NaN;
                M(:,:,k) = src;
                M2(:,:,k) = returnRoundSources(M(:,:,k));
        
                M3 = M2(:,:,k);
                i1 = find(isnan(M3)); 
                M3(i1) = Inf;
        
                subplot(3,6,(m-1)*6+k), imagesc(M3); hold on; axis equal
                axis off
                catch
                end
        
            end
        
        end
        outputFilePath = strcat(outputDir, sprintf(save_files{f}, variable));
        print(g2, '-dpdf', '-r600', outputFilePath);
        clf(g2);

    end
end

