
function M3 = returnRoundSources(M)

% figure;
% nFigs = ceil(sqrt(size(vq,3)));
% for k=1:size(vq,3),
% 
%     subplot(nFigs,nFigs,k), imagesc(vq(:,:,k)); hold on;
% 
% end
% 
% M = vq(:,:,1);


for j1 = 1:size(M,1),

    % locate borders per row
    i1 = find(~isnan(M(j1,:)));
    if (~isempty(i1))
        BordersXY(j1,:) = [ i1(1) i1(end) ];
        
    else
        BordersXY(j1,:) = [ NaN NaN ];
    end

end

CenterXY = [50.5 50.5]; %[round(nanmean(nanmean(BordersXY(:,2)-BordersXY(:,1),2))) 51 ];

myangle = -pi:0.1:pi;
RM = nan(101,101);
AngleM = nan(101,101);

% find radius at that angle
for j1 = 1:size(M,1),

    for j2 = 1:size(M,2),

        if (~isnan(M(j1,j2)))
            AngleM(j1,j2) = atan2(j1-CenterXY(1), j2-CenterXY(2));
            RM(j1,j2) = ((j1-CenterXY(1))^2 + (j2-CenterXY(2))^2)^.5;
        end            

    end
end


% reshape
R2max = 50;
M2 = nan(101,101);
for j1 = 1:size(M,1), %y

    for j2 = 1:size(M,2), %x

        if (~isnan(M(j1,j2)))

            if (CenterXY(1)~=j1 & CenterXY(2)~=j2)

                R1 = ([j1 j2] - CenterXY);
                ROrig = (R1(1)^2+R1(2)^2)^.5;
                if (ROrig>73)
                    keyboard;
                end
                alpha1 = atan2(j1-CenterXY(1), j2-CenterXY(2));

                % find points most aligned
                [yamin,xamin]=find(abs(AngleM - alpha1)<0.01);
                % find rmax for that angle
                RF = RM(yamin,xamin);
                RMax = nanmax(RF(:));

                Ri = ROrig*R2max/RMax;

                yifinal = round(CenterXY(1)+Ri*sin(alpha1));
                xifinal = round(CenterXY(2)+Ri*cos(alpha1));

                M2(yifinal,xifinal) = M(j1,j2);

            else
                 M2(j1,j2) = M(j1,j2);
            end

        end
    
    end
end

% figure; 
% imagesc(M2);

% interpolate
[X,Y] = meshgrid(1:1:101,1:1:101);
[X1,Y1] = meshgrid(1:.1:101,1:.1:101);
i1 = find(~isnan(M2(:)));
X2 = 1:.3:101;
Y2 = 1:.3:101;
[M3] = gridfit(X(:), Y(:), M2(:), X2, Y2, 'smooth',10);

CenterXY2=round(.5*[numel(X2) numel(Y2)]);
for j1 = 1:size(M3,1), %y

    for j2 = 1:size(M3,2), %x

        R1 = ([j1 j2] - CenterXY2);
        ROrig = (R1(1)^2+R1(2)^2)^.5;
        if (ROrig>.5*numel(X2))
            M3(j1,j2) = NaN;
        end
    end
end
% figure; 
% imagesc(M3);

