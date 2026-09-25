# Anxious Arcade Reels

Makes Reels / Shorts in GitHub's cloud. No laptop needed. Everything comes out 1080x1920.

## From your phone

1. Open this repository in the **GitHub app** (or github.com in the browser).
2. Tap **Actions**, pick a job, tap **Run workflow**, fill the boxes, tap **Run**.
3. When it finishes (green tick), open **Releases** on the repository page and tap the `.mp4` to download it.

| Job | What to fill in | Time |
|---|---|---|
| 1. Team fight montage | Broadcast link(s), teams as `TAG=Full name; TAG2=Full name` | 1.5-4 h per broadcast |
| 2. Day highlight Reel | Broadcast link, title lines, length | 1.5-2 h |
| 3. Add end screen + Mortals | Video link | 2-5 min |
| 4. Song hook on a video | Video link, song link | 2-5 min |

Team names are what the broadcast shows: `TAG=Team Apex Gaming; SOUL=iQOO SOUL; OG=iQOO Orangutan; GENS=Genesis Esports; RNTX=iQOO Revenant Xspark`.

For your own video (jobs 3 and 4): upload it to Google Drive, Share, "Anyone with the link", copy link, paste it in the Video box.

## If a job fails with "Sign in to confirm you're not a bot"

YouTube is blocking GitHub's servers. Add your YouTube cookies:
export `cookies.txt` for youtube.com (browser extension "Get cookies.txt LOCALLY"), then in the repository
**Settings > Secrets and variables > Actions > New repository secret**, name `YT_COOKIES`, paste the file contents.

## Limits

- One job runs at most 6 hours. Two full broadcasts in job 1 fits; more may not.
- Fights are picked from sound (gunfire and caster excitement) while the team is followed on screen.
  Watch before posting: the casters are not checked against each clip.
