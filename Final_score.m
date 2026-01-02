% --- Batch HDR-VDP-3 Evaluation ---
addpath(genpath('C:\Users\rudra\OneDrive\Desktop\Task3\hdrvdp-3.0.7'));

gt_dir    = 'C:\Users\rudra\OneDrive\Desktop\Task3\HDR_data_1024x512_test_hdr\';
recon_dir = 'C:\Users\rudra\OneDrive\Desktop\Task3\DeepHDR_Results_Final_test\';

% Collect ground truth HDRs
gt_files = dir(fullfile(gt_dir, '*.hdr'));
ppd  = 60;

fprintf('\n%-45s | %-8s | %-8s\n', 'Image ID', 'Normal_Q', 'Gamma_Q');
fprintf('%s\n', repmat('-', 1, 70));

all_normal_Q = [];
all_gamma_Q  = [];

for i = 1:numel(gt_files)
    [~, gt_name, ~] = fileparts(gt_files(i).name);
    id = gt_name;  % like 9C4A0001-db1d4a14f3

    % Find matching reconstructions
    normal_path = fullfile(recon_dir, ['pix2pix_fake_' id '.hdr']);
    gamma_path  = fullfile(recon_dir, ['pix2pix_fake_' id '_gamma.hdr']);

    if ~isfile(normal_path) || ~isfile(gamma_path)
        continue; % skip if missing
    end

    % Read all three
    gt_img     = hdrread(fullfile(gt_dir, [id '.hdr']));
    normal_img = hdrread(normal_path);
    gamma_img  = hdrread(gamma_path);

    % Resize to match GT
    if ~isequal(size(gt_img), size(normal_img))
        normal_img = imresize(normal_img, [size(gt_img,1), size(gt_img,2)]);
    end
    if ~isequal(size(gt_img), size(gamma_img))
        gamma_img = imresize(gamma_img, [size(gt_img,1), size(gt_img,2)]);
    end

    opts = {'use_gpu', false, 'disable_lowvals_warning', true, 'quiet', true};

    % Compute scores
    res_n = hdrvdp3('quality', normal_img, gt_img, 'rgb-native', ppd, opts);
    res_g = hdrvdp3('quality', gamma_img,  gt_img, 'rgb-native', ppd, opts);

    fprintf('%-45s | %8.4f | %8.4f\n', id, res_n.Q, res_g.Q);

    % store scores
    all_normal_Q(end+1) = res_n.Q;
    all_gamma_Q(end+1)  = res_g.Q;
end

% --- Averages ---
if ~isempty(all_normal_Q)
    avg_normal = mean(all_normal_Q);
    avg_gamma  = mean(all_gamma_Q);
    fprintf('%s\n', repmat('-', 1, 70));
    fprintf('%-45s | %8.4f | %8.4f\n', 'Average', avg_normal, avg_gamma);
else
    fprintf('\nNo valid image pairs found.\n');
end

fprintf('\nDone.\n');
