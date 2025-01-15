

load ('mysourcesPlot-25-T1.mat');


nFigs = ceil(sqrt(size(vq,3)));
figure;
for k=1:size(vq,3),

    M3 = returnRoundSources(vq(:,:,k));
    subplot(nFigs,nFigs,k), imagesc(M3); hold on;

end

