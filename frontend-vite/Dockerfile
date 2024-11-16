FROM node:20

WORKDIR /app/frontend-vite

COPY package.json /app/frontend-vite
RUN npm install && npm install -g serve

COPY . /app/frontend-vite
# RUN npm run build
# CMD serve -s build
CMD npm run dev