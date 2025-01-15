

clear all;

g1 = figure; 
nFig = 6;
listF = {'alpha','beta','gamma'};

cmap = jet;
cmap(end,:) = [1,1,1];
colormap(cmap);

for m=1:3,

    for k=1:6,

        try
        load(sprintf('common_sources_MG/PT_socpart/mean_best_src_PT_socpart_%s_gp%d.mat',listF{m},k-1));

        i1 = find(src==0); src(i1)=NaN;
        M(:,:,k) = src;
        M2(:,:,k) = returnRoundSources(M(:,:,k));

        M3 = M2(:,:,k);
        i1 = find(isnan(M3)); M3(i1) = Inf;
        subplot(3,6,(m-1)*6+k), imagesc(M3); hold on; axis equal
        axis off
        catch
        end

    end

end
print(g1,'-dpdf','-r600','PT_socpart.pdf'); 

g2 = figure; 
cmap = jet;
cmap(end,:) = [1,1,1];
colormap(cmap);

nFig = 6;
listF = {'alpha','beta','gamma'};
for m=1:3,

    for k=1:6,

        try
        load(sprintf('common_sources_MG/SI_socpart/mean_best_src_SI_socpart_%s_gp%d.mat',listF{m},k-1));

        i1 = find(src==0); src(i1)=NaN;
        M(:,:,k) = src;
        M2(:,:,k) = returnRoundSources(M(:,:,k));

        M3 = M2(:,:,k);
        i1 = find(isnan(M3)); M3(i1) = Inf;
        subplot(3,6,(m-1)*6+k), imagesc(M3); hold on; axis equal
        axis off
        catch
        end

    end

end
print(g2,'-dpdf','-r600','SI_socpart.pdf'); 
