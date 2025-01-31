# market analysis bot

a python-based trading bot that performs technical analysis using various indicators.
the docker image will be around 1gb when built.

## hella easy setup guide for beginners

### 1. download the code
1. click the green "code" button at the top of this page
2. click "download zip" from the dropdown menu
3. go to your downloads folder
4. find the downloaded zip file (probably called `market-analysis-bot-main.zip`)
5. right click and extract/unzip it
6. make sure the name of the extracted folder to exactly `market-analysis-bot` (extremely important!)

### 2. get docker desktop
1. go to [www.docker.com](https://www.docker.com)
2. click "download docker desktop"
3. choose your system (windows/mac)
4. install docker desktop:
   - windows: run the .exe file, use all recommended settings
   - mac: drag docker to applications folder
5. open docker desktop
6. click "sign in" and choose "sign in with github"
7. skip the survey if it appears
8. wait until you see "docker is running" with a green light, should be on the bottom left

### 3. enable docker's terminal
1. in docker desktop, find the terminal icon at bottom left
2. click it and choose "enable"
3. wait for it to finish setting up

### 4. go to your code folder
open the terminal that docker desktop just enabled and type these commands:

for windows:
```bash
cd %USERPROFILE%\Downloads\market-analysis-bot
```

for mac/linux:
```bash
cd ~/Downloads/market-analysis-bot
```

if that doesn't work, you might need to check where you extracted the folder. type:
- windows: `dir` to see files
- mac/linux: `ls` to see files

*you can use ur fav llm (chatpgt / deepseek / claude) to help you troubleshoot finding the correct `cd` command if u are struggling*

### 5. set up environment variables
you have two options for setting up your alpha vantage api key:

option 1 - using a .env file:
1. go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. sign up for a free api key
3. copy the key they give you
4. in your terminal, type (replace `YOUR_KEY_HERE` with the key you copied):
```bash
echo "ALPHA_VANTAGE_API_KEY=YOUR_KEY_HERE" > .env
```

This should work if your terminal in the market-analysis-bot directory, from your previous cd command

option 2 - you can pass the key directly when running, more annoying but still works (see run commands below)

### 6. build and run the bot

first, build the image:
```bash
docker build -t market-analysis-bot:3.13 .
```
this will take about a few minutes. you'll see lots of text - ur basically a hacker now.
when it's done, check the images tab in docker desktop, you should see `market-analysis-bot` (about 1gb).

### 7. run the bot in your native terminal (recommended, u can still use the command in docker desktop terminal if u want)

it's better to run the bot in your system's native terminal for better output viewing:

for windows:
1. press `win + r`
2. type `cmd` and press enter
3. navigate to your folder:
```bash
cd %USERPROFILE%\Downloads\market-analysis-bot
```

for mac:
1. press `cmd + space`
2. type `terminal` and press enter
3. navigate to your folder:
```bash
cd ~/Downloads/market-analysis-bot
```

then run the bot using one of these commands:

if you created a .env file:
```bash
docker run -it --rm --env-file .env market-analysis-bot:3.13
```

or pass your api key directly (if you skipped the echo .env part in step 5):
```bash
docker run -it --rm -e ALPHA_VANTAGE_API_KEY=your_key_here market-analysis-bot:3.13
```

that's it the bot should now be running

## troubleshooting

if you see "no such file or directory":
- make sure you're in the right folder
- type `pwd` to see your current folder
- use `cd` to navigate to the right place

if docker gives errors:
- make sure docker desktop is running (green light)
- try restarting docker desktop
- make sure you're signed in

if the bot doesn't work:
- check your api key in the .env file or try passing it directly
- make sure you didn't miss any steps
- try building the image again, delete in docker desktop (trash icon on the right) and start over

## for developers

we've added default environment variables in the dockerfile:
```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
ENV ALPHA_VANTAGE_API_KEY="default_value_replace_at_runtime"
```

if you want to modify the python file, do so using the GitHub website or if you have git and GitHub set up with your preferred IDE, that works fine too, just push to main as normal and ill review changes

## clean up docker stuff

to remove old containers and images:
```bash
docker container prune
docker image prune
```

or just use the trash button in docker desktop thats easier

## license

if u got here then u dont need a license lmfao