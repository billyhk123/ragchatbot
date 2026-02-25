FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY index.js ./

VOLUME /app/wa_session

EXPOSE 3001

CMD ["node", "index.js"]
