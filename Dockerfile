FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN git clone --no-checkout https://github.com/CR0ZFP/Computer-vision-and-Image-processing.git /tmp/repo && \
          cd /tmp/repo && \
          git sparse-checkout init --cone && \
          git sparse-checkout set agyikepek_4_osztaly && \
          git checkout && \
          cp -r agyikepek_4_osztaly /app/data

COPY . .
