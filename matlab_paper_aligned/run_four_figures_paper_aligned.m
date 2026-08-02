function report = run_four_figures_paper_aligned()
% Generate the paper-aligned Figures 3--6 from v0.7.0 raw results.
% Lines connect measured checkpoints only; no interpolation is applied.

root = fileparts(fileparts(mfilename('fullpath')));
deadline = readtable(fullfile(root,'results','raw', ...
    'deadline_curves_paper_aligned_50seeds.csv'));
deployment = readtable(fullfile(root,'results','raw', ...
    'deployment_scaling_paper_aligned.csv'));
bos = readtable(fullfile(root,'results','raw', ...
    'bos_scaling_paper_aligned.csv'));
validateInputs(deadline,deployment,bos);

out = fullfile(root,'results','figures_paper_aligned_matlab');
if ~exist(out,'dir'), mkdir(out); end
makeDeadline(deadline,fullfile(out,'Figure3_deadline_success.png'));
makeCostDelay(deadline,fullfile(out,'Figure4_macro_cost_delay.png'));
makeResources(deadline,fullfile(out,'Figure5_transfer_cpu_ram.png'));
makeScale(bos,fullfile(out, ...
    'Figure6_runtime_bos_acceptance.png'));

report = table(4,height(deadline),height(deployment),height(bos), ...
    'VariableNames',{'pngCount','deadlineRows', ...
    'deploymentRows','bosRows'});
disp(report);
end

function validateInputs(D,T,B)
[~,configs,~,~,~] = plotDesign();
assert(height(D)==24000,'Deadline matrix must contain 24,000 rows.');
assert(height(T)==800,'Deployment matrix must contain 800 rows.');
assert(height(B)==380,'BoS matrix must contain 380 rows.');
assert(isequal(sort(string(unique(D.configuration))),sort(configs')));
assert(isequal(sort(string(unique(T.configuration))),sort(configs')));
assert(isequal(sort(string(unique(B.configuration))),sort(configs')));
assert(all(B.first_round_submitted_sfc_count==B.controlled_bos_size), ...
    'BoS denominator must be the first submitted ready set.');
ratio = B.first_round_admitted_sfc_count ./ ...
    B.first_round_submitted_sfc_count;
assert(max(abs(ratio-B.first_round_accepted_sfc_ratio))<1e-12, ...
    'Stored first-round acceptance ratio is inconsistent.');
end

function [families,configs,colors,markers,lineStyles] = plotDesign()
families = ["montage","epigenomics","inspiral","cybershake"];
configs = ["HDS-Sharable","PBQS-Sharable", ...
    "HDS-NonSharable","PBQS-NonSharable"];
colors = [0.0000 0.4471 0.6980; 0.8353 0.3686 0.0000; ...
    0.3373 0.7059 0.9137; 0.9020 0.6235 0.0000];
markers = {'o','s','^','d'};
lineStyles = {'-','-','--','--'};
end

function makeDeadline(T,path)
[~,configs,colors,markers,lineStyles] = plotDesign();
families = ["epigenomics","cybershake"];
titles = ["(a) Epigenomics","(b) CyberShake"];
f = figure('Color','w','Position',[100 100 1200 430]);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
legendAx = gobjects(1);
configHandles = gobjects(numel(configs),1);
for i = 1:numel(families)
    ax = nexttile(tl); hold(ax,'on');
    if i==1, legendAx = ax; end
    for j = 1:numel(configs)
        R = T(string(T.family)==families(i) & ...
            string(T.configuration)==configs(j) & ...
            T.deadline_factor>=0.8 & T.deadline_factor<=3.0,:);
        x = unique(R.deadline_factor);
        globalRate = nan(size(x));
        sfcRate = nan(size(x));
        for k = 1:numel(x)
            Q = R(R.deadline_factor==x(k),:);
            globalRate(k) = 100*mean(Q.global_deadline_success);
            sfcRate(k) = 100*mean(Q.sfc_subdeadline_success);
        end
        plot(ax,x,sfcRate,':','Color',colors(j,:), ...
            'LineWidth',2.2,'HandleVisibility','off');
        h = plot(ax,x,globalRate,'Color',colors(j,:), ...
            'Marker',markers{j},'LineStyle',lineStyles{j}, ...
            'LineWidth',1.5,'MarkerSize',4, ...
            'DisplayName',configs(j));
        if i==1, configHandles(j) = h; end
    end
    title(ax,titles(i));
    xlabel(ax,'Deadline factor \kappa');
    if i==1, ylabel(ax,'Success rate (%)'); end
    xlim(ax,[0.8 3.0]); ylim(ax,[-3 103]);
    xticks(ax,[0.8 1.0 1.5 2.0 2.5 3.0]);
    grid(ax,'on');
end
globalHandle = plot(legendAx,nan,nan,'-','Color',[0.2 0.2 0.2], ...
    'LineWidth',1.5,'DisplayName','Global deadline');
sfcHandle = plot(legendAx,nan,nan,':','Color',[0.2 0.2 0.2], ...
    'LineWidth',1.8,'DisplayName','All SFC subdeadlines');
legend(legendAx,[configHandles;globalHandle;sfcHandle], ...
    [cellstr(configs),{'Global deadline','All SFC subdeadlines'}], ...
    'Location','southeast','NumColumns',2,'Box','on');
savePng(f,path);
end

function makeCostDelay(T,path)
[~,configs,colors,markers,lineStyles] = plotDesign();
metrics = ["provisioning_cost","end_to_end_delay_s"];
ylabels = ["Provisioning cost ($)","End-to-end delay (s)"];
titles = ["(a) Macro-average provisioning cost", ...
    "(b) Macro-average end-to-end delay"];
f = figure('Color','w','Position',[100 100 1260 400]);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
legendHandles = gobjects(numel(configs),1);
legendAx = gobjects(1);
for m = 1:2
    ax = nexttile(tl); hold(ax,'on');
    firstValid = inf; lastValid = -inf;
    for j = 1:numel(configs)
        [x,y] = macroMetricCurve(T,configs(j),metrics(m),5);
        if isempty(x), continue; end
        firstValid = min(firstValid,min(x));
        lastValid = max(lastValid,max(x));
        h = plot(ax,x,y,'Color',colors(j,:),'Marker',markers{j}, ...
            'LineStyle',lineStyles{j},'LineWidth',1.5, ...
            'MarkerSize',4,'DisplayName',configs(j));
        if m==2, legendHandles(j) = h; end
    end
    formatMacroAxis(ax,firstValid,lastValid,2.0);
    title(ax,titles(m)); xlabel(ax,'Deadline factor \kappa');
    ylabel(ax,ylabels(m)); grid(ax,'on');
    if m==2, legendAx = ax; end
end
legend(legendAx,legendHandles,cellstr(configs), ...
    'Location','best','NumColumns',2,'Box','on');
savePng(f,path);
end

function makeResources(T,path)
[~,configs,colors,markers,lineStyles] = plotDesign();
f = figure('Color','w','Position',[100 100 1260 400]);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

ax = nexttile(tl); hold(ax,'on');
firstValid = inf; lastValid = -inf;
algorithmHandles = gobjects(numel(configs),1);
for j = 1:numel(configs)
    [x,y] = macroMetricCurve(T,configs(j),"network_data_mb",5);
    firstValid = min(firstValid,min(x)); lastValid = max(lastValid,max(x));
    algorithmHandles(j) = plot(ax,x,y,'Color',colors(j,:), ...
        'Marker',markers{j}, ...
        'LineStyle',lineStyles{j},'LineWidth',1.5, ...
        'MarkerSize',4,'DisplayName',configs(j));
end
formatMacroAxis(ax,firstValid,lastValid,2.0);
title(ax,'(a) Transferred data'); xlabel(ax,'Deadline factor \kappa');
ylabel(ax,'Transferred data (MB)'); grid(ax,'on');
legend(ax,algorithmHandles,cellstr(configs), ...
    'Location','best','NumColumns',2,'Box','on');

ax = nexttile(tl); hold(ax,'on');
firstValid = inf; lastValid = -inf;
for j = 1:numel(configs)
    [x,cpu] = macroMetricCurve(T,configs(j),"cpu_utilization",5);
    [~,ram] = macroMetricCurve(T,configs(j),"ram_utilization",5);
    firstValid = min(firstValid,min(x)); lastValid = max(lastValid,max(x));
    plot(ax,x,100*cpu,'Color',colors(j,:),'Marker',markers{j}, ...
        'MarkerFaceColor',colors(j,:),'LineStyle',lineStyles{j}, ...
        'LineWidth',1.5,'MarkerSize',4,'DisplayName',configs(j));
    plot(ax,x,100*ram,'Color',colors(j,:),'Marker',markers{j}, ...
        'MarkerFaceColor','w','LineStyle',lineStyles{j}, ...
        'LineWidth',1.2,'MarkerSize',4,'HandleVisibility','off');
end
formatMacroAxis(ax,firstValid,lastValid,2.0);
title(ax,'(b) CPU and RAM utilization');
xlabel(ax,'Deadline factor \kappa');
ylabel(ax,'Purchased-capacity utilization (%)'); grid(ax,'on');
cpuHandle = plot(ax,nan,nan,'o','LineStyle','none', ...
    'Color',[0.2 0.2 0.2],'MarkerFaceColor',[0.2 0.2 0.2]);
ramHandle = plot(ax,nan,nan,'o','LineStyle','none', ...
    'Color',[0.2 0.2 0.2],'MarkerFaceColor','w');
legend(ax,[cpuHandle;ramHandle],{'CPU (filled)','RAM (open)'}, ...
    'Location','best','NumColumns',2,'Box','on');
savePng(f,path);
end

function makeScale(B,path)
[~,configs,colors,markers,lineStyles] = plotDesign();
f = figure('Color','w','Position',[60 60 1080 410]);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

ax = nexttile(tl); hold(ax,'on');
algorithmHandles = gobjects(numel(configs),1);
for j = 1:numel(configs)
    R = B(string(B.configuration)==configs(j),:);
    x = unique(R.controlled_bos_size); y = nan(size(x));
    for k = 1:numel(x)
        Q = R(R.controlled_bos_size==x(k),:);
        y(k) = median(Q.first_round_scheduler_runtime_s,'omitnan');
    end
    algorithmHandles(j) = plot(ax,x,y,'Color',colors(j,:), ...
        'Marker',markers{j}, ...
        'LineStyle',lineStyles{j},'LineWidth',1.5, ...
        'MarkerSize',4,'DisplayName',configs(j));
end
capacity = median(B.initial_effective_candidate_count,'omitnan');
xline(ax,capacity,':','Color',[0.4 0.4 0.4], ...
    'HandleVisibility','off');
set(ax,'YScale','log'); xticks(ax,2:2:20);
xlabel(ax,'Ready SFCs per BoS');
ylabel(ax,'Median scheduling runtime (s)');
title(ax,'(a) BoS-width runtime'); grid(ax,'on');

ax = nexttile(tl); hold(ax,'on');
for j = 1:numel(configs)
    R = B(string(B.configuration)==configs(j),:);
    x = unique(R.controlled_bos_size); y = nan(size(x));
    for k = 1:numel(x)
        Q = R(R.controlled_bos_size==x(k),:);
        y(k) = 100*mean(Q.first_round_accepted_sfc_ratio,'omitnan');
    end
    plot(ax,x,y,'Color',colors(j,:),'Marker',markers{j}, ...
        'LineStyle',lineStyles{j},'LineWidth',1.5, ...
        'MarkerSize',4,'DisplayName',configs(j));
end
xline(ax,capacity,':','Color',[0.4 0.4 0.4], ...
    'HandleVisibility','off');
ylim(ax,[-3 103]); xticks(ax,2:2:20);
xlabel(ax,'Ready SFCs per BoS'); ylabel(ax,'Accepted-SFC ratio (%)');
title(ax,'(b) BoS accepted-SFC ratio'); grid(ax,'on');
legend(ax,algorithmHandles,cellstr(configs), ...
    'Location','southwest','NumColumns',2,'Box','on');

savePng(f,path);
end

function formatMacroAxis(ax,firstValid,lastValid,lower)
if nargin<4
    lower = max(0.8,firstValid-0.25);
end
xlim(ax,[lower lastValid+0.04]);
ticks = [0.8 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0];
xticks(ax,ticks(ticks>=lower & ticks<=lastValid));
if firstValid>lower
    patch(ax,[lower firstValid firstValid lower], ...
        [0 0 max(ylim(ax)) max(ylim(ax))],[0.75 0.75 0.75], ...
        'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
end
if lastValid>3
    xline(ax,3,'--','Color',[0.4 0.4 0.4], ...
        'HandleVisibility','off');
end
end

function [x,y] = macroMetricCurve(T,config,metric,minN)
[families,~,~,~,~] = plotDesign();
P = commonDeadlineSuccessRows(T);
P = P(string(P.configuration)==config,:);
xAll = unique(P.deadline_factor);
x = []; y = [];
for k = 1:numel(xAll)
    familyMeans = nan(numel(families),1);
    valid = true;
    for i = 1:numel(families)
        Q = P(P.deadline_factor==xAll(k) & ...
            string(P.family)==families(i),:);
        values = Q.(metric);
        values = values(isfinite(values));
        if numel(values)<minN
            valid = false; break;
        end
        familyMeans(i) = mean(values);
    end
    if valid
        x(end+1,1) = xAll(k); %#ok<AGROW>
        y(end+1,1) = mean(familyMeans); %#ok<AGROW>
    end
end
end

function P = commonDeadlineSuccessRows(T)
[~,configs,~,~,~] = plotDesign();
keys = deadlineKey(T(string(T.configuration)==configs(1) & ...
    T.global_deadline_success==1,:));
for i = 2:numel(configs)
    Q = T(string(T.configuration)==configs(i) & ...
        T.global_deadline_success==1,:);
    keys = intersect(keys,deadlineKey(Q));
end
P = T(T.global_deadline_success==1 & ...
    ismember(deadlineKey(T),keys),:);
end

function keys = deadlineKey(T)
keys = string(T.family)+"|"+string(T.workflow_size)+"|"+ ...
    string(T.deadline_factor)+"|"+string(T.seed);
end

function savePng(f,path)
drawnow;
print(f,path,'-dpng','-r300');
close(f);
end
