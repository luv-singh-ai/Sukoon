# inspired from https://stackoverflow.com/a/41597992
FROM node:20

WORKDIR /app/sukoon-frontend

COPY package.json /app/sukoon-frontend
RUN npm install && npm install -g serve

COPY . /app/sukoon-frontend
# RUN npm run build
# CMD serve -s build
CMD npm start
