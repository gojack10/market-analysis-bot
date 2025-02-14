# Use Python 3.13 RC slim image as base
FROM python:3.13-rc-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    autotools-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Build & install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && cp /usr/share/misc/config.guess . \
    && cp /usr/share/misc/config.sub . \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Verify where TA-Lib is installed
RUN ls -la /usr/include/ta-lib || echo "TA-Lib headers not in /usr/include/ta-lib" \
    && ls -la /usr/local/include/ta-lib || echo "TA-Lib headers not in /usr/local/include/ta-lib" \
    && ls -la /usr/lib/libta_lib* || ls -la /usr/local/lib/libta_lib* || echo "TA-Lib library not found"

# Create symlink (if needed) and update linker cache
RUN if [ -f /usr/lib/libta_lib.so ]; then \
      ln -s /usr/lib/libta_lib.so /usr/lib/libta-lib.so; \
    elif [ -f /usr/local/lib/libta_lib.so ]; then \
      ln -s /usr/local/lib/libta_lib.so /usr/local/lib/libta-lib.so; \
    fi \
    && ldconfig

# Set TA-Lib environment variables for the Python wrapper
# (Adjust these if your verification step shows headers/libraries in /usr/local)
ENV TA_LIBRARY_PATH=/usr/local/lib \
    TA_INCLUDE_PATH=/usr/local/include

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (ensure TA_* vars are active during build)
RUN TA_LIBRARY_PATH=/usr/local/lib TA_INCLUDE_PATH=/usr/local/include pip install --no-cache-dir -v ta-lib && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Command to run your Python script
CMD ["python", "trade_server.py"]
