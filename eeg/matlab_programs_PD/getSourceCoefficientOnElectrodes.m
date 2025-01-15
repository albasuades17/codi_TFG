function sourceCM = getSourceCoefficientOnElectrodes(cfgic, varargin, idSubject, basicDir)

ic_data2 = varargin;
fprintf('idSubject %d \n', idSubject);


% get positions for all electrodes
%cfg.layout = ft_prepare_layout(cfg, data);

cfgp = [];
cfgp.layout = ft_prepare_layout(cfgic, ic_data2);

% select layout by label
selcomp = cfgic.component;

xyelectrodes = zeros(numel(selcomp),2);
nT = 1;
for k=1:numel(selcomp)

    for q=1:numel(cfgp.layout.label)

        if (strcmp(cfgp.layout.label{q}, ic_data2.topolabel{k})==1)
           xyelectrodes(nT,:) = cfgp.layout.pos(q,:);
           nT = nT + 1;
        end
    end
end

ztopo = (zscore(ic_data2.topo));

topostd = nanstd(ztopo(:));%ic_data2.topo(:));
topomean = nanmean(ztopo(:));%ic_data2.topo(:));
% for each source
for k=1:numel(selcomp),
    
    % calculate center of mass
    mytopo = zeros(size(ic_data2.topo(:,k)));
%     iplus2 = ic_data2.topo(:,k)>topostd;
%     iminus2 = ic_data2.topo(:,k)<topostd;
    iplus2 = ztopo(:,k)>topostd;
    iminus2 = ztopo(:,k)<-topostd;
    
%         xys(:,1,k) = xyelectrodes(:,1).*(ic_data2.topo(i2,k)); % coefficients per electrode for this component
%         xys(:,2,k) = xyelectrodes(:,2).*(ic_data2.topo(i2,k));
%    if (nansum(ic_data2.topo(iplus2,k))>nansum(ic_data2.topo(iminus2,k)))            
    if (nansum(iplus2)>nansum(iminus2)) 
        xys(:,1,k) = xyelectrodes(:,1).*iplus2;
        xys(:,2,k) = xyelectrodes(:,2).*iplus2;
        sourceCM(k,1) = nansum(xys(:,1,k))/nansum(iplus2);
        sourceCM(k,2) = nansum(xys(:,2,k))/nansum(iplus2);
    else
        mytopo(iminus2) = ztopo(iminus2,k);        
        xys(:,1,k) = xyelectrodes(:,1).*iminus2;
        xys(:,2,k) = xyelectrodes(:,2).*iminus2;
        sourceCM(k,1) = nansum(xys(:,1,k))/nansum(iminus2);
        sourceCM(k,2) = nansum(xys(:,2,k))/nansum(iminus2);
    end    
    
end

if (1)
    g1 = figure;
    plot(xyelectrodes(:,1), xyelectrodes(:,2),'bo'); hold on;
    plot(sourceCM(:,1), sourceCM(:,2),'r.','markersize',10);
    set(g1,'color',[1 1 1]);
    print(g1,'-dpdf','-r300','centerOfMass.pdf');

    % plot each component
    g2 = figure;
    xaxis = nanmin(xyelectrodes(:,1)):(nanmax(xyelectrodes(:,1))-nanmin(xyelectrodes(:,1)))/100:nanmax(xyelectrodes(:,1));
    yaxis = nanmin(xyelectrodes(:,2)):(nanmax(xyelectrodes(:,2))-nanmin(xyelectrodes(:,2)))/100:nanmax(xyelectrodes(:,2));
    [xq,yq] = meshgrid(xaxis,yaxis);

    nplots = ceil(sqrt(size(ic_data2.topo,2)));
    for k=1:numel(selcomp),

        %plot3(xyelectrodes(:,1),xyelectrodes(:,2),ic_data2.topo(:,k),'o'); 
        vq(:,:,k) = griddata(xyelectrodes(:,1), xyelectrodes(:,2),ic_data2.topo(:,k),xq,yq);
        %mesh(xq,yq,vq)
        subplot(nplots,nplots,k),imagesc(flipud(vq(:,:,k)));
        hold on
        set(gca,'fontname','Arial','fontsize',7);
        
    %    plot3(xyelectrodes(:,1), xyelectrodes(:,2),ic_data2.topo(:,k),'o')

    end
    
    set(g2, 'Position', get(0, 'Screensize'));

    set(g2,'color',[1 1 1]);
    print(g2,'-dpdf','-r300','-bestfit',sprintf('%s/mysources-%d-T1-2.pdf',basicDir,idSubject));
    
    save(sprintf('%s/mysourcesPlot3-%d-T1.mat',basicDir,idSubject),'vq');
    clear('vq');
end

